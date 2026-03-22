"""
================================================================================
BirdCLEF+ 2026 - Training Script v4 (Simplified & Effective)
================================================================================

Key Learnings from v3 Failure:
  1. MLP HURTS performance (LR=0.918, MLP=0.852) → DROP MLP
  2. Pseudo-labeling adds noise → DROP or be very conservative  
  3. train_audio dilutes PCA → Train PCA on soundscapes ONLY
  4. More data ≠ Better when distribution differs

v4 Strategy:
  - Use ONLY LR probes (drop MLP entirely)
  - NO pseudo-labeling (too noisy)
  - PCA trained on soundscapes only
  - Sample weighting for soundscape data
  - Keep train_audio with lower weight

Output Files:
  - full_perch_arrays.npz
  - full_perch_meta.parquet

"""

# =============================================================================
# INSTALL TENSORFLOW 2.20 (Required for Perch v2 GPU)
# =============================================================================
!pip install -q --no-deps /kaggle/input/notebooks/kdmitrie/bc26-tensorflow-2-20-0/wheel/tensorboard-2.20.0-py3-none-any.whl
!pip install -q --no-deps /kaggle/input/notebooks/kdmitrie/bc26-tensorflow-2-20-0/wheel/tensorflow-2.20.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gc
import json
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from tqdm.auto import tqdm
import pickle

warnings.filterwarnings("ignore")
tf.experimental.numpy.experimental_enable_numpy_behavior()

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE = Path("/kaggle/input/competitions/birdclef-2026")
MODEL_DIR = Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2/2")
OUTPUT_DIR = Path("/kaggle/working")

# Audio parameters
SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12
BATCH_FILES = 4

# Feature flags
USE_TRAIN_AUDIO = True
USE_PSEUDO_LABELING = False  # DISABLED - adds noise, hurts LB

# =============================================================================
# v4 SIMPLIFIED PARAMETERS
# =============================================================================
# Higher PCA dimensions, trained on soundscapes only
BEST_PROBE = {
    "pca_dim": 128,          # Increased from 64
    "min_pos": 8,
    "C": 0.50,
    "alpha": 0.40,
}

# No ensemble needed - using LR only
BEST_FUSION = {
    "lambda_event": 0.4,
    "lambda_texture": 1.0,
    "lambda_proxy_texture": 0.8,
    "smooth_texture": 0.35,
}

# Sample weights - prioritize soundscape data
SAMPLE_WEIGHTS = {
    "soundscape_weight": 10.0,   # Soundscapes weighted 10x
    "train_audio_weight": 1.0,   # Train audio normal weight
}

print("=" * 70)
print("BirdCLEF+ 2026 - Training v4 (Simplified)")
print("=" * 70)
print(f"USE_TRAIN_AUDIO: {USE_TRAIN_AUDIO}")
print(f"USE_PSEUDO_LABELING: {USE_PSEUDO_LABELING}")
print(f"PCA dimensions: {BEST_PROBE['pca_dim']}")
print(f"Model: LR ONLY (MLP dropped - was hurting performance)")

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n" + "-" * 70)
print("Loading competition data...")
print("-" * 70)

taxonomy = pd.read_csv(BASE / "taxonomy.csv")
sample_sub = pd.read_csv(BASE / "sample_submission.csv")
soundscape_labels = pd.read_csv(BASE / "train_soundscapes_labels.csv")
train_audio_meta = pd.read_csv(BASE / "train.csv")

soundscape_labels = soundscape_labels.drop_duplicates().reset_index(drop=True)

PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES = len(PRIMARY_LABELS)
label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}

taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
soundscape_labels["primary_label"] = soundscape_labels["primary_label"].astype(str)
train_audio_meta["primary_label"] = train_audio_meta["primary_label"].astype(str)

print(f"Classes: {N_CLASSES}")
print(f"train.csv rows: {len(train_audio_meta)}")
print(f"Labeled soundscape windows: {len(soundscape_labels)}")

# =============================================================================
# PARSE LABELS & FILENAMES
# =============================================================================
def parse_labels(x):
    if pd.isna(x):
        return []
    return [t.strip() for t in str(x).split(";") if t.strip()]

def union_labels(series):
    return sorted(set(lbl for x in series for lbl in parse_labels(x)))

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")

def parse_filename(name):
    m = FNAME_RE.match(name)
    if not m:
        return {"site": None, "hour_utc": -1}
    _, site, _, hms = m.groups()
    return {"site": site, "hour_utc": int(hms[:2])}

sc_clean = soundscape_labels.groupby(["filename", "start", "end"])["primary_label"].apply(union_labels).reset_index(name="label_list")
sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
sc_clean["row_id"] = sc_clean["filename"].str.replace(".ogg", "", regex=False) + "_" + sc_clean["end_sec"].astype(str)

meta = sc_clean["filename"].apply(parse_filename).apply(pd.Series)
sc_clean = pd.concat([sc_clean, meta], axis=1)

all_labeled_files = sorted(sc_clean["filename"].unique().tolist())
all_soundscape_files = sorted([f.name for f in (BASE / "train_soundscapes").glob("*.ogg")])
unlabeled_files = sorted(set(all_soundscape_files) - set(all_labeled_files))

print(f"Labeled soundscape files: {len(all_labeled_files)}")
print(f"Unlabeled soundscape files: {len(unlabeled_files)}")

Y_SC = np.zeros((len(sc_clean), N_CLASSES), dtype=np.uint8)
for i, labels in enumerate(sc_clean["label_list"]):
    for lbl in labels:
        if lbl in label_to_idx:
            Y_SC[i, label_to_idx[lbl]] = 1

print(f"Label density in soundscapes: {Y_SC.mean():.4%}")

# =============================================================================
# PERCH MODEL LOADING (GPU)
# =============================================================================
print("\n" + "-" * 70)
print("Loading Perch v2 model (GPU)...")
print("-" * 70)

birdclassifier = tf.saved_model.load(str(MODEL_DIR))
infer_fn = birdclassifier.signatures["serving_default"]

bc_labels = pd.read_csv(MODEL_DIR / "assets" / "labels.csv").reset_index()
bc_labels = bc_labels.rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
NO_LABEL_INDEX = len(bc_labels)

print(f"Perch model loaded. Classes: {len(bc_labels)}")

# =============================================================================
# PERCH LABEL MAPPING
# =============================================================================
print("\nMapping competition labels to Perch labels...")

bc_lookup = bc_labels[["scientific_name", "bc_index"]].copy()
mapping = taxonomy.merge(bc_lookup, on="scientific_name", how="left")
mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL_INDEX).astype(int)

label_to_bc_index = mapping.set_index("primary_label")["bc_index"]
BC_INDICES = np.array([int(label_to_bc_index.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)

MAPPED_MASK = BC_INDICES != NO_LABEL_INDEX
MAPPED_POS = np.where(MAPPED_MASK)[0].astype(np.int32)
UNMAPPED_POS = np.where(~MAPPED_MASK)[0].astype(np.int32)
MAPPED_BC_INDICES = BC_INDICES[MAPPED_MASK].astype(np.int32)

print(f"Mapped: {MAPPED_MASK.sum()}, Unmapped: {(~MAPPED_MASK).sum()}")

CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
TEXTURE_TAXA = {"Amphibia", "Insecta"}

ACTIVE_CLASSES = [PRIMARY_LABELS[i] for i in np.where(Y_SC.sum(axis=0) > 0)[0]]

idx_active_texture = np.array([label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) in TEXTURE_TAXA], dtype=np.int32)
idx_active_event = np.array([label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) not in TEXTURE_TAXA], dtype=np.int32)

idx_mapped_active_texture = idx_active_texture[np.isin(idx_active_texture, MAPPED_POS)]
idx_mapped_active_event = idx_active_event[np.isin(idx_active_event, MAPPED_POS)]
idx_unmapped_active_texture = idx_active_texture[~np.isin(idx_active_texture, MAPPED_POS)]
idx_unmapped_active_event = idx_active_event[~np.isin(idx_active_event, MAPPED_POS)]
idx_unmapped_inactive = np.array([i for i in UNMAPPED_POS if PRIMARY_LABELS[i] not in ACTIVE_CLASSES], dtype=np.int32)

# =============================================================================
# FROG PROXIES
# =============================================================================
print("\nBuilding frog proxies for unmapped amphibians...")

unmapped_df = mapping[mapping["bc_index"] == NO_LABEL_INDEX].copy()
unmapped_non_sonotype = unmapped_df[~unmapped_df["primary_label"].astype(str).str.contains("son", na=False)].copy()

def get_genus_hits(sci_name):
    genus = str(sci_name).split()[0]
    return bc_labels[bc_labels["scientific_name"].astype(str).str.match(rf"^{re.escape(genus)}\s", na=False)]

proxy_map = {}
for _, row in unmapped_non_sonotype.iterrows():
    hits = get_genus_hits(row["scientific_name"])
    if len(hits) > 0:
        proxy_map[row["primary_label"]] = hits["bc_index"].astype(int).tolist()

SELECTED_PROXY_TARGETS = sorted([t for t in proxy_map.keys() if CLASS_NAME_MAP.get(t) == "Amphibia"])
selected_proxy_pos = np.array([label_to_idx[c] for c in SELECTED_PROXY_TARGETS], dtype=np.int32)
selected_proxy_pos_to_bc = {label_to_idx[t]: np.array(proxy_map[t], dtype=np.int32) for t in SELECTED_PROXY_TARGETS}

idx_selected_proxy_active_texture = np.intersect1d(selected_proxy_pos, idx_active_texture)
idx_selected_prioronly_active_texture = np.setdiff1d(idx_unmapped_active_texture, selected_proxy_pos)
idx_selected_prioronly_active_event = np.setdiff1d(idx_unmapped_active_event, selected_proxy_pos)

print(f"Texture classes: {len(idx_active_texture)}, Event classes: {len(idx_active_event)}")
print(f"Frog proxies: {len(SELECTED_PROXY_TARGETS)}")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def macro_auc(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    if keep.sum() == 0:
        return 0.0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def smooth_cols(scores, cols, alpha=0.35):
    if alpha <= 0 or len(cols) == 0:
        return scores.copy()
    s = scores.copy()
    n_samples = s.shape[0]
    if n_samples % N_WINDOWS != 0:
        return s
    view = s.reshape(-1, N_WINDOWS, s.shape[1])
    x = view[:, :, cols]
    prev_x = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    next_x = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    view[:, :, cols] = (1.0 - alpha) * x + 0.5 * alpha * (prev_x + next_x)
    return s

def seq_features_1d(v):
    n_samples = len(v)
    if n_samples % N_WINDOWS != 0:
        return v.copy(), v.copy(), v.copy(), v.copy()
    x = v.reshape(-1, N_WINDOWS)
    return (
        np.concatenate([x[:, :1], x[:, :-1]], axis=1).reshape(-1),
        np.concatenate([x[:, 1:], x[:, -1:]], axis=1).reshape(-1),
        np.repeat(x.mean(axis=1), N_WINDOWS),
        np.repeat(x.max(axis=1), N_WINDOWS),
    )

def build_class_features(emb, raw, prior, base):
    prev_b, next_b, mean_b, max_b = seq_features_1d(base)
    return np.concatenate([
        emb, raw[:, None], prior[:, None], base[:, None],
        prev_b[:, None], next_b[:, None], mean_b[:, None], max_b[:, None]
    ], axis=1).astype(np.float32)

# =============================================================================
# PRIOR TABLES
# =============================================================================
def fit_prior_tables(df, Y):
    df = df.reset_index(drop=True)
    global_p = Y.mean(axis=0).astype(np.float32)
    
    site_keys = sorted(df["site"].dropna().astype(str).unique().tolist())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_n = np.zeros(len(site_keys), dtype=np.float32)
    site_p = np.zeros((len(site_keys), Y.shape[1]), dtype=np.float32)
    for s in site_keys:
        m = df["site"].astype(str).values == s
        site_n[site_to_i[s]] = m.sum()
        site_p[site_to_i[s]] = Y[m].mean(axis=0)
    
    hour_keys = sorted(df["hour_utc"].dropna().astype(int).unique().tolist())
    hour_to_i = {h: i for i, h in enumerate(hour_keys)}
    hour_n = np.zeros(len(hour_keys), dtype=np.float32)
    hour_p = np.zeros((len(hour_keys), Y.shape[1]), dtype=np.float32)
    for h in hour_keys:
        m = df["hour_utc"].astype(int).values == h
        hour_n[hour_to_i[h]] = m.sum()
        hour_p[hour_to_i[h]] = Y[m].mean(axis=0)
    
    sh_to_i = {}
    sh_n_list, sh_p_list = [], []
    for (s, h), idx in df.groupby(["site", "hour_utc"]).groups.items():
        sh_to_i[(str(s), int(h))] = len(sh_n_list)
        idx = np.array(list(idx))
        sh_n_list.append(len(idx))
        sh_p_list.append(Y[idx].mean(axis=0))
    
    return {
        "global_p": global_p,
        "site_to_i": site_to_i, "site_n": site_n, "site_p": site_p,
        "hour_to_i": hour_to_i, "hour_n": hour_n, "hour_p": hour_p,
        "sh_to_i": sh_to_i,
        "sh_n": np.array(sh_n_list, dtype=np.float32),
        "sh_p": np.stack(sh_p_list).astype(np.float32) if sh_p_list else np.zeros((0, Y.shape[1]), dtype=np.float32),
    }

def prior_logits(sites, hours, tables, eps=1e-4):
    n = len(sites)
    p = np.repeat(tables["global_p"][None, :], n, axis=0).astype(np.float32, copy=True)
    
    site_idx = np.fromiter((tables["site_to_i"].get(str(s), -1) for s in sites), dtype=np.int32, count=n)
    hour_idx = np.fromiter((tables["hour_to_i"].get(int(h), -1) if int(h) >= 0 else -1 for h in hours), dtype=np.int32, count=n)
    sh_idx = np.fromiter((tables["sh_to_i"].get((str(s), int(h)), -1) if int(h) >= 0 else -1 for s, h in zip(sites, hours)), dtype=np.int32, count=n)
    
    valid = hour_idx >= 0
    if valid.any():
        nh = tables["hour_n"][hour_idx[valid]][:, None]
        p[valid] = nh / (nh + 8.0) * tables["hour_p"][hour_idx[valid]] + (1.0 - nh / (nh + 8.0)) * p[valid]
    
    valid = site_idx >= 0
    if valid.any():
        ns = tables["site_n"][site_idx[valid]][:, None]
        p[valid] = ns / (ns + 8.0) * tables["site_p"][site_idx[valid]] + (1.0 - ns / (ns + 8.0)) * p[valid]
    
    valid = sh_idx >= 0
    if valid.any():
        nsh = tables["sh_n"][sh_idx[valid]][:, None]
        p[valid] = nsh / (nsh + 4.0) * tables["sh_p"][sh_idx[valid]] + (1.0 - nsh / (nsh + 4.0)) * p[valid]
    
    np.clip(p, eps, 1.0 - eps, out=p)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)

def fuse_scores(base, sites, hours, tables):
    scores = base.copy()
    prior = prior_logits(sites, hours, tables)
    
    if len(idx_mapped_active_event):
        scores[:, idx_mapped_active_event] += BEST_FUSION["lambda_event"] * prior[:, idx_mapped_active_event]
    if len(idx_mapped_active_texture):
        scores[:, idx_mapped_active_texture] += BEST_FUSION["lambda_texture"] * prior[:, idx_mapped_active_texture]
    if len(idx_selected_proxy_active_texture):
        scores[:, idx_selected_proxy_active_texture] += BEST_FUSION["lambda_proxy_texture"] * prior[:, idx_selected_proxy_active_texture]
    if len(idx_selected_prioronly_active_event):
        scores[:, idx_selected_prioronly_active_event] = BEST_FUSION["lambda_event"] * prior[:, idx_selected_prioronly_active_event]
    if len(idx_selected_prioronly_active_texture):
        scores[:, idx_selected_prioronly_active_texture] = BEST_FUSION["lambda_texture"] * prior[:, idx_selected_prioronly_active_texture]
    if len(idx_unmapped_inactive):
        scores[:, idx_unmapped_inactive] = -8.0
    
    scores = smooth_cols(scores, idx_active_texture, alpha=BEST_FUSION["smooth_texture"])
    return scores.astype(np.float32), prior

# =============================================================================
# AUDIO PROCESSING
# =============================================================================
def read_audio(path, target_samples=FILE_SAMPLES):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    elif len(y) > target_samples:
        y = y[:target_samples]
    return y

def read_soundscape_batch(paths):
    def read_one(path):
        try:
            return read_audio(path)
        except:
            return np.zeros(FILE_SAMPLES, dtype=np.float32)
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(read_one, paths))
    return np.stack(results, axis=0).astype(np.float32)

def load_single_audio(path):
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if len(y) < WINDOW_SAMPLES:
            y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
        elif len(y) > WINDOW_SAMPLES:
            center = (len(y) - WINDOW_SAMPLES) // 2
            y = y[center:center + WINDOW_SAMPLES]
        return y
    except:
        return np.zeros(WINDOW_SAMPLES, dtype=np.float32)

def load_audio_batch_parallel(paths, n_workers=8):
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(load_single_audio, paths))
    return np.stack(results, axis=0).astype(np.float32)

# =============================================================================
# PERCH INFERENCE
# =============================================================================
def infer_perch_soundscapes(paths, verbose=True):
    paths = [Path(p) for p in paths]
    n_files = len(paths)
    n_rows = n_files * N_WINDOWS
    
    row_ids = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    hours = np.empty(n_rows, dtype=np.int16)
    scores = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    embeddings = np.zeros((n_rows, 1536), dtype=np.float32)
    
    write_row = 0
    iterator = range(0, n_files, BATCH_FILES)
    if verbose:
        iterator = tqdm(iterator, total=(n_files + BATCH_FILES - 1) // BATCH_FILES, desc="Perch Soundscapes")
    
    for start in iterator:
        batch_paths = paths[start:start + BATCH_FILES]
        batch_size = len(batch_paths)
        batch_start = write_row
        
        audio_batch = read_soundscape_batch(batch_paths)
        x = audio_batch.reshape(batch_size * N_WINDOWS, WINDOW_SAMPLES)
        
        for path in batch_paths:
            meta = parse_filename(path.name)
            row_ids[write_row:write_row + N_WINDOWS] = [f"{path.stem}_{t}" for t in range(5, 65, 5)]
            sites[write_row:write_row + N_WINDOWS] = meta["site"]
            hours[write_row:write_row + N_WINDOWS] = int(meta["hour_utc"])
            write_row += N_WINDOWS
        
        with tf.device('/GPU:0'):
            outputs = infer_fn(inputs=tf.convert_to_tensor(x))
            logits = outputs["label"].numpy().astype(np.float32)
            emb = outputs["embedding"].numpy().astype(np.float32)
        
        scores[batch_start:write_row, MAPPED_POS] = logits[:write_row - batch_start, MAPPED_BC_INDICES]
        embeddings[batch_start:write_row] = emb
        
        for pos, bc_idx_arr in selected_proxy_pos_to_bc.items():
            scores[batch_start:write_row, pos] = logits[:write_row - batch_start, bc_idx_arr].max(axis=1)
        
        del x, outputs, logits, emb
        tf.keras.backend.clear_session()
        gc.collect()
    
    return pd.DataFrame({"row_id": row_ids, "site": sites, "hour_utc": hours}), scores, embeddings

def infer_perch_short(paths, labels, verbose=True):
    paths = [Path(p) for p in paths]
    n_files = len(paths)
    
    scores = np.zeros((n_files, N_CLASSES), dtype=np.float32)
    embeddings = np.zeros((n_files, 1536), dtype=np.float32)
    
    batch_size = BATCH_FILES * N_WINDOWS
    iterator = range(0, n_files, batch_size)
    if verbose:
        iterator = tqdm(iterator, total=(n_files + batch_size - 1) // batch_size, desc="Perch Short")
    
    for start in iterator:
        batch_paths = paths[start:start + batch_size]
        n_batch = len(batch_paths)
        
        x = load_audio_batch_parallel(batch_paths, n_workers=8)
        
        with tf.device('/GPU:0'):
            outputs = infer_fn(inputs=tf.convert_to_tensor(x))
            logits = outputs["label"].numpy().astype(np.float32)
            emb = outputs["embedding"].numpy().astype(np.float32)
        
        end = start + n_batch
        scores[start:end, MAPPED_POS] = logits[:, MAPPED_BC_INDICES]
        embeddings[start:end] = emb
        
        for pos, bc_idx_arr in selected_proxy_pos_to_bc.items():
            scores[start:end, pos] = logits[:, bc_idx_arr].max(axis=1)
        
        del x, outputs, logits, emb
        tf.keras.backend.clear_session()
        gc.collect()
    
    Y = np.zeros((n_files, N_CLASSES), dtype=np.uint8)
    for i, lbl in enumerate(labels):
        if lbl in label_to_idx:
            Y[i, label_to_idx[lbl]] = 1
    
    return scores, embeddings, Y

# =============================================================================
# RUN PERCH ON LABELED SOUNDSCAPES
# =============================================================================
print("\n" + "-" * 70)
print("Running Perch on labeled training soundscapes...")
print("-" * 70)

all_labeled_paths = [BASE / "train_soundscapes" / fn for fn in all_labeled_files]
meta_all, scores_raw, emb_all = infer_perch_soundscapes(all_labeled_paths)

labeled_row_ids = set(sc_clean["row_id"].tolist())
keep_mask = meta_all["row_id"].isin(labeled_row_ids).values

meta_all = meta_all[keep_mask].reset_index(drop=True)
scores_raw = scores_raw[keep_mask]
emb_all = emb_all[keep_mask]

row_id_to_label_idx = {row_id: idx for idx, row_id in enumerate(sc_clean["row_id"])}
Y_TRAIN = np.zeros((len(meta_all), N_CLASSES), dtype=np.uint8)
for i, row_id in enumerate(meta_all["row_id"]):
    Y_TRAIN[i] = Y_SC[row_id_to_label_idx[row_id]]

print(f"Total windows processed: {len(meta_all)}")
print(f"scores_raw: {scores_raw.shape}, emb_all: {emb_all.shape}")
print(f"Y_TRAIN: {Y_TRAIN.shape}, positive rate: {Y_TRAIN.mean():.4%}")

# =============================================================================
# RUN PERCH ON TRAIN_AUDIO (OPTIONAL)
# =============================================================================
if USE_TRAIN_AUDIO:
    print("\n" + "-" * 70)
    print("Running Perch on train_audio (short clips)...")
    print("-" * 70)
    
    valid_audio = train_audio_meta[train_audio_meta["primary_label"].isin(PRIMARY_LABELS)].copy()
    print(f"Valid train_audio files: {len(valid_audio)}")
    
    audio_paths = [BASE / "train_audio" / row["filename"] for _, row in valid_audio.iterrows()]
    audio_labels = valid_audio["primary_label"].tolist()
    
    CHUNK_SIZE = 5000
    scores_audio_list = []
    emb_audio_list = []
    Y_audio_list = []
    
    for chunk_start in range(0, len(audio_paths), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(audio_paths))
        chunk_paths = audio_paths[chunk_start:chunk_end]
        chunk_labels = audio_labels[chunk_start:chunk_end]
        
        scores_chunk, emb_chunk, Y_chunk = infer_perch_short(chunk_paths, chunk_labels)
        scores_audio_list.append(scores_chunk)
        emb_audio_list.append(emb_chunk)
        Y_audio_list.append(Y_chunk)
        
        print(f"  Processed {chunk_end}/{len(audio_paths)} files")
    
    scores_audio = np.concatenate(scores_audio_list, axis=0)
    emb_audio = np.concatenate(emb_audio_list, axis=0)
    Y_audio = np.concatenate(Y_audio_list, axis=0)
    
    print(f"\ntrain_audio: {len(scores_audio)} clips")
else:
    scores_audio = np.zeros((0, N_CLASSES), dtype=np.float32)
    emb_audio = np.zeros((0, 1536), dtype=np.float32)
    Y_audio = np.zeros((0, N_CLASSES), dtype=np.uint8)
    print("\nSkipping train_audio")

# =============================================================================
# BUILD OOF (Soundscapes only)
# =============================================================================
print("\nBuilding OOF (soundscapes only)...")

gkf = GroupKFold(n_splits=5)
groups = meta_all["site"].to_numpy()
oof_base = np.zeros_like(scores_raw, dtype=np.float32)
oof_prior = np.zeros_like(scores_raw, dtype=np.float32)

for _, va_idx in tqdm(list(gkf.split(scores_raw, groups=groups)), desc="OOF"):
    va_idx = np.sort(va_idx)
    val_sites = set(meta_all.iloc[va_idx]["site"].tolist())
    prior_m = ~sc_clean["site"].isin(val_sites).values
    tables = fit_prior_tables(sc_clean.loc[prior_m].reset_index(drop=True), Y_SC[prior_m])
    oof_base[va_idx], oof_prior[va_idx] = fuse_scores(
        scores_raw[va_idx],
        meta_all.iloc[va_idx]["site"].to_numpy(),
        meta_all.iloc[va_idx]["hour_utc"].to_numpy(),
        tables
    )

print(f"\nOOF AUC (soundscapes): {macro_auc(Y_TRAIN, oof_base):.6f}")

audio_prior = np.zeros_like(scores_audio, dtype=np.float32) if len(scores_audio) > 0 else np.zeros((0, N_CLASSES), dtype=np.float32)
audio_base = scores_audio.copy() if len(scores_audio) > 0 else np.zeros((0, N_CLASSES), dtype=np.float32)

# =============================================================================
# v4: DOMAIN-AWARE PCA - Train on Soundscapes ONLY
# =============================================================================
print("\n" + "-" * 70)
print("Training LR probes (v4: Soundscape-only PCA)...")
print("-" * 70)

# Fit scaler and PCA on SOUNSCAPE embeddings only
emb_scaler = StandardScaler()
emb_scaler.fit(emb_all)

emb_all_scaled = emb_scaler.transform(emb_all)
if len(emb_audio) > 0:
    emb_audio_scaled = emb_scaler.transform(emb_audio)
else:
    emb_audio_scaled = np.zeros((0, 1536), dtype=np.float32)

# Fit PCA on soundscapes only
n_comp = min(int(BEST_PROBE["pca_dim"]), len(emb_all) - 1, emb_all.shape[1])
emb_pca = PCA(n_components=n_comp)
emb_pca.fit(emb_all_scaled)

print(f"PCA trained on SOUNSCAPE embeddings only")
print(f"PCA: {n_comp} components, variance: {emb_pca.explained_variance_ratio_.sum():.4f}")

# Transform all data
Z_soundscape = emb_pca.transform(emb_all_scaled).astype(np.float32)
if len(emb_audio_scaled) > 0:
    Z_audio = emb_pca.transform(emb_audio_scaled).astype(np.float32)
else:
    Z_audio = np.zeros((0, n_comp), dtype=np.float32)

# Combine
Z_combined = np.concatenate([Z_soundscape, Z_audio], axis=0) if len(Z_audio) > 0 else Z_soundscape
scores_combined = np.concatenate([scores_raw, scores_audio], axis=0) if len(scores_audio) > 0 else scores_raw
Y_COMBINED = np.concatenate([Y_TRAIN, Y_audio], axis=0) if len(Y_audio) > 0 else Y_TRAIN
prior_combined = np.concatenate([oof_prior, audio_prior], axis=0) if len(audio_prior) > 0 else oof_prior
base_combined = np.concatenate([oof_base, audio_base], axis=0) if len(audio_base) > 0 else oof_base

print(f"Combined training data: {len(Z_combined)} samples")

# =============================================================================
# SAMPLE WEIGHTS
# =============================================================================
print("\nApplying sample weights...")

n_soundscapes = len(Z_soundscape)
n_train_audio = len(Z_audio) if len(Z_audio) > 0 else 0

sample_weights = np.ones(len(Z_combined), dtype=np.float32)
sample_weights[:n_soundscapes] = SAMPLE_WEIGHTS["soundscape_weight"]
if n_train_audio > 0:
    sample_weights[n_soundscapes:] = SAMPLE_WEIGHTS["train_audio_weight"]

print(f"Soundscape samples: {n_soundscapes} (weight={SAMPLE_WEIGHTS['soundscape_weight']}x)")
print(f"Train audio samples: {n_train_audio} (weight={SAMPLE_WEIGHTS['train_audio_weight']}x)")

# =============================================================================
# TRAIN LR PROBES ONLY (No MLP - it was hurting performance)
# =============================================================================
pos_counts = Y_COMBINED.sum(axis=0)
probe_idx = np.where(pos_counts >= int(BEST_PROBE["min_pos"]))[0].astype(np.int32)

lr_probes = {}
for cls_idx in tqdm(probe_idx, desc="LR Probes"):
    y = Y_COMBINED[:, cls_idx]
    if y.sum() == 0 or y.sum() == len(y):
        continue
    X = build_class_features(Z_combined, scores_combined[:, cls_idx], prior_combined[:, cls_idx], base_combined[:, cls_idx])
    clf = LogisticRegression(C=float(BEST_PROBE["C"]), max_iter=400, solver="liblinear", class_weight="balanced")
    clf.fit(X, y, sample_weight=sample_weights)
    lr_probes[cls_idx] = clf

print(f"\nTrained {len(lr_probes)} LR probes (MLP skipped - was hurting performance)")

# =============================================================================
# EVALUATE OOF
# =============================================================================
print("\n" + "-" * 70)
print("Evaluating OOF with LR probes...")
print("-" * 70)

Z_oof = Z_soundscape

oof_lr = oof_base.copy()
for cls_idx, clf in tqdm(lr_probes.items(), desc="LR OOF", leave=False):
    X = build_class_features(Z_oof, scores_raw[:, cls_idx], oof_prior[:, cls_idx], oof_base[:, cls_idx])
    pred = clf.decision_function(X).astype(np.float32)
    oof_lr[:, cls_idx] = (1.0 - BEST_PROBE["alpha"]) * oof_base[:, cls_idx] + BEST_PROBE["alpha"] * pred

print("\n--- Soundscape OOF Results ---")
print(f"OOF AUC (base):  {macro_auc(Y_TRAIN, oof_base):.6f}")
print(f"OOF AUC (LR):    {macro_auc(Y_TRAIN, sigmoid(oof_lr)):.6f}")

# =============================================================================
# SAVE ARTIFACTS
# =============================================================================
print("\n" + "-" * 70)
print("Saving artifacts...")
print("-" * 70)

def serialize_pickle(obj):
    buf = BytesIO()
    pickle.dump(obj, buf)
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)

np.savez_compressed(
    OUTPUT_DIR / "full_perch_arrays.npz",
    embeddings=emb_all.astype(np.float32),
    scores_raw=scores_raw.astype(np.float32),
    oof_base=oof_base.astype(np.float32),
    oof_prior=oof_prior.astype(np.float32),
    oof_lr=oof_lr.astype(np.float32),
    emb_audio=emb_audio.astype(np.float32) if len(emb_audio) > 0 else np.zeros((0, 1536), dtype=np.float32),
    scores_audio=scores_audio.astype(np.float32) if len(scores_audio) > 0 else np.zeros((0, N_CLASSES), dtype=np.float32),
    bc_indices=BC_INDICES,
    mapped_pos=MAPPED_POS,
    unmapped_pos=UNMAPPED_POS,
    mapped_bc_indices=MAPPED_BC_INDICES,
    selected_proxy_pos=selected_proxy_pos,
    idx_active_texture=idx_active_texture,
    idx_active_event=idx_active_event,
    idx_mapped_active_texture=idx_mapped_active_texture,
    idx_mapped_active_event=idx_mapped_active_event,
    idx_selected_proxy_active_texture=idx_selected_proxy_active_texture,
    idx_selected_prioronly_active_texture=idx_selected_prioronly_active_texture,
    idx_selected_prioronly_active_event=idx_selected_prioronly_active_event,
    idx_unmapped_inactive=idx_unmapped_inactive,
    proxy_map_keys=np.array(list(selected_proxy_pos_to_bc.keys()), dtype=np.int32),
    proxy_map_vals=np.array([np.array(selected_proxy_pos_to_bc[k], dtype=np.int32) for k in selected_proxy_pos_to_bc.keys()], dtype=object),
    config=np.array(json.dumps({
        "best_probe": BEST_PROBE,
        "best_ensemble": {"method": "none", "lr_weight": 1.0, "mlp_weight": 0.0},  # LR only
        "best_fusion": BEST_FUSION,
        "sample_weights": SAMPLE_WEIGHTS,
        "n_classes": N_CLASSES,
        "n_windows": N_WINDOWS,
        "n_soundscapes": len(emb_all),
        "n_train_audio": len(emb_audio),
        "use_pseudo_labeling": False,
        "use_train_audio": USE_TRAIN_AUDIO,
    })),
    labels=np.array(PRIMARY_LABELS, dtype=object),
    lr_probes_bytes=serialize_pickle(lr_probes),
    mlp_probes_bytes=serialize_pickle({}),  # Empty - no MLP
    emb_scaler_bytes=serialize_pickle(emb_scaler),
    emb_pca_bytes=serialize_pickle(emb_pca),
    prior_tables_bytes=serialize_pickle(fit_prior_tables(sc_clean.reset_index(drop=True), Y_SC)),
)

meta_all.to_parquet(OUTPUT_DIR / "full_perch_meta.parquet", index=False)

print("\n" + "=" * 70)
print("Training Complete!")
print("=" * 70)
print(f"\nData Summary:")
print(f"  Soundscapes (labeled):  {len(emb_all)} windows")
print(f"  Train Audio:            {len(emb_audio)} clips")
print(f"\nOutput Files:")
print(f"  - full_perch_arrays.npz")
print(f"  - full_perch_meta.parquet")
print(f"\nMetrics:")
print(f"  OOF AUC (LR): {macro_auc(Y_TRAIN, sigmoid(oof_lr)):.6f}")