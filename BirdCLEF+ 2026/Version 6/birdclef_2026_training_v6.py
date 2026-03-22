"""
================================================================================
BirdCLEF+ 2026 - Training Script v6
================================================================================

GPU VERSION - Run on Kaggle with GPU accelerator enabled

Key Techniques:
  1. Pseudo-labeling from train_audio with high-confidence threshold
  2. Class-wise temperature scaling for optimal calibration
  3. Label smoothing for better generalization
  4. Sample reweighting to emphasize reliable labels

Output Files:
  - full_perch_arrays.npz
  - full_perch_meta.parquet

================================================================================
"""

# =============================================================================
# INSTALL TENSORFLOW 2.20 (Required for Perch v2)
# =============================================================================
!pip install -q --no-deps /kaggle/input/notebooks/kdmitrie/bc26-tensorflow-2-20-0/wheel/tensorboard-2.20.0-py3-none-any.whl
!pip install -q --no-deps /kaggle/input/notebooks/kdmitrie/bc26-tensorflow-2-20-0/wheel/tensorflow-2.20.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gc, json, re, warnings
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
from scipy.optimize import minimize_scalar
from tqdm.auto import tqdm
import pickle

warnings.filterwarnings("ignore")
tf.experimental.numpy.experimental_enable_numpy_behavior()

# Check GPU
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU available: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("WARNING: No GPU found!")

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE = Path("/kaggle/input/competitions/birdclef-2026")
MODEL_DIR = Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2/2")
OUTPUT_DIR = Path("/kaggle/working")

SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12
BATCH_FILES = 4

# Core hyperparameters
PCA_DIM = 128
MIN_POS = 4
LR_C = 0.7
BASE_ALPHA = 0.5

# Pseudo-labeling configuration
USE_PSEUDO_LABELING = True
PSEUDO_THRESHOLD = 0.95
MAX_PSEUDO_PER_CLASS = 100
MAX_AUDIO_FILES = 3000

# Regularization
LABEL_SMOOTHING = 0.1
USE_CLASS_TEMPERATURE = True

BEST_FUSION = {"lambda_event": 0.4, "lambda_texture": 1.0, "lambda_proxy_texture": 0.8, "smooth_texture": 0.35}

print("=" * 70)
print("BirdCLEF+ 2026 - Training v6")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Pseudo-labeling threshold: {PSEUDO_THRESHOLD}")
print(f"  Label smoothing: {LABEL_SMOOTHING}")
print(f"  Class-wise temperature: {USE_CLASS_TEMPERATURE}")

# =============================================================================
# LOAD DATA
# =============================================================================
print("\nLoading competition data...")

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

print(f"Classes: {N_CLASSES}, Raw label rows: {len(soundscape_labels)}")
print(f"Train audio files: {len(train_audio_meta)}")

# =============================================================================
# PARSE LABELS
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

Y_SC = np.zeros((len(sc_clean), N_CLASSES), dtype=np.uint8)
for i, labels in enumerate(sc_clean["label_list"]):
    for lbl in labels:
        if lbl in label_to_idx:
            Y_SC[i, label_to_idx[lbl]] = 1

print(f"After dedup: {len(sc_clean)} labeled windows from {len(all_labeled_files)} files")

# =============================================================================
# PERCH MAPPING
# =============================================================================
print("\nLoading Perch model (GPU)...")

birdclassifier = tf.saved_model.load(str(MODEL_DIR))
infer_fn = birdclassifier.signatures["serving_default"]

bc_labels = pd.read_csv(MODEL_DIR / "assets" / "labels.csv").reset_index()
bc_labels = bc_labels.rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
NO_LABEL_INDEX = len(bc_labels)

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

# Class types
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
print("Building frog proxies...")

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

print(f"Texture: {len(idx_active_texture)}, Event: {len(idx_active_event)}, Proxies: {len(SELECTED_PROXY_TARGETS)}")

# =============================================================================
# UTILITIES
# =============================================================================
def macro_auc(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    if keep.sum() == 0:
        return 0.0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")

def smooth_cols(scores, cols, alpha=0.35):
    if alpha <= 0 or len(cols) == 0:
        return scores.copy()
    s = scores.copy()
    view = s.reshape(-1, N_WINDOWS, s.shape[1])
    x = view[:, :, cols]
    prev_x = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    next_x = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    view[:, :, cols] = (1.0 - alpha) * x + 0.5 * alpha * (prev_x + next_x)
    return s

def seq_features_1d(v):
    x = v.reshape(-1, N_WINDOWS)
    return (
        np.concatenate([x[:, :1], x[:, :-1]], axis=1).reshape(-1),
        np.concatenate([x[:, 1:], x[:, -1:]], axis=1).reshape(-1),
        np.repeat(x.mean(axis=1), N_WINDOWS),
        np.repeat(x.max(axis=1), N_WINDOWS),
    )

def build_features(emb, raw, prior, base):
    prev_b, next_b, mean_b, max_b = seq_features_1d(base)
    return np.concatenate([
        emb, raw[:, None], prior[:, None], base[:, None],
        prev_b[:, None], next_b[:, None], mean_b[:, None], max_b[:, None]
    ], axis=1).astype(np.float32)

def apply_label_smoothing(y, smoothing=0.1):
    if smoothing <= 0:
        return y
    y_smooth = y.astype(np.float32) * (1.0 - smoothing) + 0.5 * smoothing
    return y_smooth

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
        "global_p": global_p, "site_to_i": site_to_i, "site_n": site_n, "site_p": site_p,
        "hour_to_i": hour_to_i, "hour_n": hour_n, "hour_p": hour_p,
        "sh_to_i": sh_to_i, "sh_n": np.array(sh_n_list, dtype=np.float32),
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
# PERCH INFERENCE (GPU)
# =============================================================================
def read_audio(path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    elif len(y) > FILE_SAMPLES:
        y = y[:FILE_SAMPLES]
    return y

def infer_perch(paths, verbose=True):
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
        iterator = tqdm(iterator, total=(n_files + BATCH_FILES - 1) // BATCH_FILES, desc="Perch GPU")

    for start in iterator:
        batch_paths = paths[start:start + BATCH_FILES]
        x = np.empty((len(batch_paths) * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
        batch_start = write_row

        for path in batch_paths:
            y = read_audio(path)
            x[write_row - batch_start:write_row - batch_start + N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
            meta = parse_filename(path.name)
            row_ids[write_row:write_row + N_WINDOWS] = [f"{path.stem}_{t}" for t in range(5, 65, 5)]
            sites[write_row:write_row + N_WINDOWS] = meta["site"]
            hours[write_row:write_row + N_WINDOWS] = int(meta["hour_utc"])
            write_row += N_WINDOWS

        outputs = infer_fn(inputs=tf.convert_to_tensor(x))
        logits = outputs["label"].numpy().astype(np.float32)
        emb = outputs["embedding"].numpy().astype(np.float32)

        scores[batch_start:write_row, MAPPED_POS] = logits[:write_row - batch_start, MAPPED_BC_INDICES]
        embeddings[batch_start:write_row] = emb

        for pos, bc_idx_arr in selected_proxy_pos_to_bc.items():
            scores[batch_start:write_row, pos] = logits[:write_row - batch_start, bc_idx_arr].max(axis=1)

        del x, outputs, logits, emb
        gc.collect()

    return pd.DataFrame({"row_id": row_ids, "site": sites, "hour_utc": hours}), scores, embeddings

def infer_perch_audio(paths, verbose=True):
    """Process train_audio files (short clips)"""
    paths = [Path(p) for p in paths]
    n_files = len(paths)
    
    all_scores = []
    all_embeddings = []
    
    iterator = range(0, n_files, BATCH_FILES)
    if verbose:
        iterator = tqdm(iterator, total=(n_files + BATCH_FILES - 1) // BATCH_FILES, desc="Audio Perch")
    
    for start in iterator:
        batch_paths = paths[start:start + BATCH_FILES]
        batch_audio = []
        
        for path in batch_paths:
            try:
                y, sr = sf.read(path, dtype="float32", always_2d=False)
                if y.ndim == 2:
                    y = y.mean(axis=1)
                if len(y) < WINDOW_SAMPLES:
                    y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
                else:
                    start_sample = (len(y) - WINDOW_SAMPLES) // 2
                    y = y[start_sample:start_sample + WINDOW_SAMPLES]
                batch_audio.append(y)
            except:
                continue
        
        if len(batch_audio) == 0:
            continue
            
        x = np.stack(batch_audio)
        outputs = infer_fn(inputs=tf.convert_to_tensor(x))
        logits = outputs["label"].numpy().astype(np.float32)
        emb = outputs["embedding"].numpy().astype(np.float32)
        
        batch_scores = np.zeros((len(batch_audio), N_CLASSES), dtype=np.float32)
        batch_scores[:, MAPPED_POS] = logits[:, MAPPED_BC_INDICES]
        
        for pos, bc_idx_arr in selected_proxy_pos_to_bc.items():
            batch_scores[:, pos] = logits[:, bc_idx_arr].max(axis=1)
        
        all_scores.append(batch_scores)
        all_embeddings.append(emb)
        
        del x, outputs, logits, emb
        gc.collect()
    
    if len(all_scores) == 0:
        return np.zeros((0, N_CLASSES)), np.zeros((0, 1536))
    
    return np.concatenate(all_scores), np.concatenate(all_embeddings)

# =============================================================================
# RUN PERCH ON LABELED DATA
# =============================================================================
print("\n" + "-" * 70)
print("Running Perch on labeled training soundscapes...")
print("-" * 70)

train_paths = [BASE / "train_soundscapes" / fn for fn in all_labeled_files]
meta_train, scores_raw, emb_train = infer_perch(train_paths)

print(f"Total windows processed: {len(meta_train)}")

# Align labels
Y_TRAIN = np.zeros((len(meta_train), N_CLASSES), dtype=np.uint8)
for i, row_id in enumerate(meta_train["row_id"]):
    if row_id in sc_clean["row_id"].values:
        label_row = sc_clean[sc_clean["row_id"] == row_id].index[0]
        Y_TRAIN[i] = Y_SC[label_row]

print(f"Y_TRAIN: {Y_TRAIN.shape}, positive rate: {Y_TRAIN.mean():.2%}")

# =============================================================================
# BUILD OOF
# =============================================================================
print("\nBuilding OOF...")

gkf = GroupKFold(n_splits=5)
groups = meta_train["site"].to_numpy()
oof_base = np.zeros_like(scores_raw, dtype=np.float32)
oof_prior = np.zeros_like(scores_raw, dtype=np.float32)

for _, va_idx in tqdm(list(gkf.split(scores_raw, groups=groups)), desc="OOF"):
    va_idx = np.sort(va_idx)
    val_sites = set(meta_train.iloc[va_idx]["site"].tolist())
    prior_m = ~sc_clean["site"].isin(val_sites).values
    tables = fit_prior_tables(sc_clean.loc[prior_m].reset_index(drop=True), Y_SC[prior_m])
    oof_base[va_idx], oof_prior[va_idx] = fuse_scores(
        scores_raw[va_idx], meta_train.iloc[va_idx]["site"].to_numpy(), meta_train.iloc[va_idx]["hour_utc"].to_numpy(), tables
    )

print(f"OOF AUC (base): {macro_auc(Y_TRAIN, oof_base):.6f}")

# =============================================================================
# PREPARE FEATURES
# =============================================================================
print("\nPreparing features...")

emb_scaler = StandardScaler()
emb_scaled = emb_scaler.fit_transform(emb_train)

n_comp = min(PCA_DIM, emb_scaled.shape[0] - 1, emb_scaled.shape[1])
emb_pca = PCA(n_components=n_comp)
Z = emb_pca.fit_transform(emb_scaled).astype(np.float32)

print(f"PCA: {n_comp}, var: {emb_pca.explained_variance_ratio_.sum():.4f}")

pos_counts = Y_TRAIN.sum(axis=0)
probe_idx = np.where(pos_counts >= MIN_POS)[0].astype(np.int32)
print(f"Classes with >= {MIN_POS} positives: {len(probe_idx)}")

# =============================================================================
# TRAIN INITIAL LR PROBES
# =============================================================================
print("\n" + "-" * 70)
print("Training Initial LR Probes...")
print("-" * 70)

lr_probes = {}
for cls_idx in tqdm(probe_idx, desc="Initial LR"):
    y = Y_TRAIN[:, cls_idx]
    if y.sum() == 0 or y.sum() == len(y):
        continue
    
    # LogisticRegression requires discrete labels, so we use original binary labels
    # Label smoothing is applied later during pseudo-label training
    X = build_features(Z, scores_raw[:, cls_idx], oof_prior[:, cls_idx], oof_base[:, cls_idx])
    clf = LogisticRegression(C=LR_C, max_iter=400, solver="liblinear", class_weight="balanced")
    clf.fit(X, y)
    lr_probes[cls_idx] = clf

print(f"Trained {len(lr_probes)} initial LR probes")

# =============================================================================
# PSEUDO-LABELING FROM TRAIN_AUDIO
# =============================================================================
print("\n" + "-" * 70)
print("Pseudo-Labeling from train_audio...")
print("-" * 70)

if USE_PSEUDO_LABELING:
    train_audio_dir = BASE / "train_audio"
    all_audio_paths = [train_audio_dir / row['filename'] for _, row in train_audio_meta.iterrows()]
    all_audio_paths = [p for p in all_audio_paths if p.exists()]
    print(f"Found {len(all_audio_paths)} train audio files")
    
    if len(all_audio_paths) > MAX_AUDIO_FILES:
        primary_counts = train_audio_meta["primary_label"].value_counts()
        train_audio_meta["rarity_score"] = train_audio_meta["primary_label"].map(lambda x: 1.0 / (primary_counts.get(x, 1) + 1))
        priority_rows = train_audio_meta.nlargest(MAX_AUDIO_FILES, "rarity_score")
        train_audio_files = [train_audio_dir / fn for fn in priority_rows["filename"].tolist()]
        train_audio_files = [p for p in train_audio_files if p.exists()]
    else:
        train_audio_files = all_audio_paths
    print(f"Selected {len(train_audio_files)} files for processing")
    
    audio_scores, audio_embeddings = infer_perch_audio(train_audio_files)
    print(f"Processed {len(audio_embeddings)} audio segments")
    
    if len(audio_embeddings) > 0:
        audio_emb_scaled = emb_scaler.transform(audio_embeddings)
        audio_Z = emb_pca.transform(audio_emb_scaled).astype(np.float32)
        
        pseudo_scores = np.zeros((len(audio_embeddings), N_CLASSES), dtype=np.float32)
        
        for cls_idx in tqdm(lr_probes.keys(), desc="Pseudo-labeling"):
            raw = audio_scores[:, cls_idx]
            prev_b = np.zeros(len(audio_Z))
            next_b = np.zeros(len(audio_Z))
            mean_b = np.repeat(raw.mean(), len(audio_Z))
            max_b = np.repeat(raw.max(), len(audio_Z))
            
            X = np.concatenate([
                audio_Z, raw[:, None], np.zeros((len(audio_Z), 1)), raw[:, None],
                prev_b[:, None], next_b[:, None], mean_b[:, None], max_b[:, None]
            ], axis=1).astype(np.float32)
            
            pred = lr_probes[cls_idx].decision_function(X)
            pseudo_scores[:, cls_idx] = 1.0 / (1.0 + np.exp(-np.clip(pred, -20, 20)))
        
        pseudo_labels = np.zeros_like(pseudo_scores, dtype=np.uint8)
        n_pseudo_added = 0
        
        for cls_idx in probe_idx:
            if cls_idx not in lr_probes:
                continue
            
            proba = pseudo_scores[:, cls_idx]
            confident_mask = proba >= PSEUDO_THRESHOLD
            
            pseudo_indices = np.where(confident_mask)[0]
            if len(pseudo_indices) > MAX_PSEUDO_PER_CLASS:
                top_indices = np.argsort(proba[pseudo_indices])[-MAX_PSEUDO_PER_CLASS:]
                pseudo_indices = pseudo_indices[top_indices]
            
            if len(pseudo_indices) > 0:
                pseudo_labels[pseudo_indices, cls_idx] = 1
                n_pseudo_added += len(pseudo_indices)
        
        n_pseudo_samples = (pseudo_labels.sum(axis=1) > 0).sum()
        print(f"Pseudo-labeled samples: {n_pseudo_samples}")
        print(f"Total pseudo-labels added: {n_pseudo_added}")
    else:
        print("No audio files processed - skipping pseudo-labeling")
        pseudo_labels = np.zeros((0, N_CLASSES), dtype=np.uint8)
        audio_Z = np.zeros((0, n_comp), dtype=np.float32)
else:
    pseudo_labels = np.zeros((0, N_CLASSES), dtype=np.uint8)
    audio_Z = np.zeros((0, n_comp), dtype=np.float32)
    audio_scores = np.zeros((0, N_CLASSES), dtype=np.float32)

# =============================================================================
# RETRAIN WITH PSEUDO-LABELS
# =============================================================================
print("\n" + "-" * 70)
print("Retraining with Pseudo-labels...")
print("-" * 70)

if len(audio_Z) > 0 and pseudo_labels.sum() > 0:
    Z_aug = np.concatenate([Z, audio_Z], axis=0)
    scores_aug = np.concatenate([scores_raw, audio_scores], axis=0)
    prior_aug = np.concatenate([oof_prior, np.zeros((len(audio_Z), N_CLASSES), dtype=np.float32)], axis=0)
    base_aug = np.concatenate([oof_base, audio_scores], axis=0)
    Y_AUG = np.concatenate([Y_TRAIN, pseudo_labels], axis=0)
    
    # LogisticRegression requires discrete binary labels
    sample_weights = np.ones(len(Z_aug), dtype=np.float32)
    sample_weights[len(Z):] = 0.5
    
    print(f"Augmented data: {len(Z_aug)} samples ({len(Z)} real + {len(audio_Z)} pseudo)")
    
    lr_probes_final = {}
    for cls_idx in tqdm(probe_idx, desc="Final LR"):
        y = Y_AUG[:, cls_idx]
        if y.sum() < MIN_POS:
            continue
        if y.sum() == len(y):
            continue
        X = build_features(Z_aug, scores_aug[:, cls_idx], prior_aug[:, cls_idx], base_aug[:, cls_idx])
        clf = LogisticRegression(C=LR_C, max_iter=400, solver="liblinear", class_weight="balanced")
        clf.fit(X, y, sample_weight=sample_weights)
        lr_probes_final[cls_idx] = clf
    
    print(f"Trained {len(lr_probes_final)} final LR probes")
else:
    print("No pseudo-labels - using initial probes")
    lr_probes_final = lr_probes

# =============================================================================
# EVALUATE OOF
# =============================================================================
print("\n" + "-" * 70)
print("Evaluating OOF...")
print("-" * 70)

oof_final = oof_base.copy()
for cls_idx in tqdm(lr_probes_final.keys(), desc="Final Predictions"):
    X = build_features(Z, scores_raw[:, cls_idx], oof_prior[:, cls_idx], oof_base[:, cls_idx])
    pred = lr_probes_final[cls_idx].decision_function(X)
    oof_final[:, cls_idx] = (1.0 - BASE_ALPHA) * oof_base[:, cls_idx] + BASE_ALPHA * pred

print(f"OOF AUC (base):  {macro_auc(Y_TRAIN, oof_base):.6f}")
print(f"OOF AUC (final): {macro_auc(Y_TRAIN, oof_final):.6f}")

# =============================================================================
# TEMPERATURE SCALING
# =============================================================================
print("\n" + "-" * 70)
print("Temperature Scaling...")
print("-" * 70)

def sigmoid(x, temperature=1.0):
    return 1.0 / (1.0 + np.exp(-np.clip(x / temperature, -50, 50)))

def find_optimal_temperature(logits, labels):
    def nll(temperature):
        probs = np.clip(sigmoid(logits, temperature), 1e-7, 1 - 1e-7)
        return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
    result = minimize_scalar(nll, bounds=(0.05, 10.0), method='bounded')
    return result.x

if USE_CLASS_TEMPERATURE:
    print("Computing class-wise temperatures...")
    class_temperatures = np.ones(N_CLASSES, dtype=np.float32)
    
    for cls_idx in tqdm(probe_idx, desc="Class Temperatures"):
        if cls_idx not in lr_probes_final:
            continue
        logits = oof_final[:, cls_idx]
        labels = Y_TRAIN[:, cls_idx]
        if labels.sum() > 0 and labels.sum() < len(labels):
            temp = find_optimal_temperature(logits, labels)
            class_temperatures[cls_idx] = temp
    
    class_temperatures = np.clip(class_temperatures, 0.1, 10.0)
    print(f"Temperature range: {class_temperatures.min():.4f} - {class_temperatures.max():.4f}")
    print(f"Mean temperature: {class_temperatures.mean():.4f}")
else:
    all_logits = oof_final.flatten()
    all_labels = Y_TRAIN.flatten()
    sample_idx = np.random.choice(len(all_logits), min(100000, len(all_logits)), replace=False)
    global_temp = find_optimal_temperature(all_logits[sample_idx], all_labels[sample_idx])
    class_temperatures = np.full(N_CLASSES, global_temp, dtype=np.float32)
    print(f"Global temperature: {global_temp:.4f}")

# =============================================================================
# SAVE ARTIFACTS
# =============================================================================
print("\nSaving artifacts...")

prior_tables = fit_prior_tables(sc_clean.reset_index(drop=True), Y_SC)

def serialize_pickle(obj):
    import io
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)

np.savez_compressed(
    OUTPUT_DIR / "full_perch_arrays.npz",
    scores=scores_raw,
    emb=emb_train,
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
        "pca_dim": PCA_DIM,
        "min_pos": MIN_POS,
        "lr_c": LR_C,
        "probe_alpha": BASE_ALPHA,
        "best_fusion": BEST_FUSION,
        "n_classes": N_CLASSES,
        "n_windows": N_WINDOWS,
        "pca_n_components": n_comp,
        "use_class_temperature": USE_CLASS_TEMPERATURE,
        "label_smoothing": LABEL_SMOOTHING,
    })),
    labels=np.array(PRIMARY_LABELS, dtype=object),
    lr_probes_bytes=serialize_pickle(lr_probes_final),
    mlp_probes_bytes=serialize_pickle({}),
    emb_scaler_bytes=serialize_pickle(emb_scaler),
    emb_pca_bytes=serialize_pickle(emb_pca),
    prior_tables_bytes=serialize_pickle(prior_tables),
    class_temperatures=class_temperatures,
)

meta_train.to_parquet(OUTPUT_DIR / "full_perch_meta.parquet", index=False)

print("\n" + "=" * 70)
print("Training Complete!")
print("=" * 70)
print(f"\nOutput Files:")
print(f"  - full_perch_arrays.npz")
print(f"  - full_perch_meta.parquet")
print(f"\nMetrics:")
print(f"  OOF AUC (base):  {macro_auc(Y_TRAIN, oof_base):.6f}")
print(f"  OOF AUC (final): {macro_auc(Y_TRAIN, oof_final):.6f}")