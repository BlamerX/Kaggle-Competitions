"""
================================================================================
BirdCLEF+ 2026 - Training Script v3 (GPU + Advanced Features)
================================================================================

Features:
  - GPU Perch v2 model (10x faster inference)
  - Train audio integration (USE_TRAIN_AUDIO = True)
  - Pseudo-labeling for unlabeled soundscapes (USE_PSEUDO_LABELING = True)
  - Min reduction ensemble (2024 1st place strategy)
  - Prior tables (site/hour distributions)
  - LR + MLP ensemble probes with v2 retraining

Output Files:
  - full_perch_arrays.npz    → embeddings + scores + models + indices
  - full_perch_meta.parquet  → metadata

"""

# =============================================================================
# INSTALL TENSORFLOW 2.20 (Required for Perch v2 GPU)
# =============================================================================
!pip install -q --no-deps /kaggle/input/notebooks/kdmitrie/bc26-tensorflow-2-20-0/wheel/tensorboard-2.20.0-py3-none-any.whl
!pip install -q --no-deps /kaggle/input/notebooks/kdmitrie/bc26-tensorflow-2-20-0/wheel/tensorflow-2.20.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# NOTE: Do NOT set CUDA_VISIBLE_DEVICES="" - we WANT GPU for training!

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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from tqdm.auto import tqdm
import joblib
import pickle

warnings.filterwarnings("ignore")
tf.experimental.numpy.experimental_enable_numpy_behavior()

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE = Path("/kaggle/input/competitions/birdclef-2026")
# GPU Perch model path (requires TensorFlow >= 2.20 and GPU)
MODEL_DIR = Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2/2")
OUTPUT_DIR = Path("/kaggle/working")

# Audio parameters
SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12
BATCH_FILES = 4  # Reduced for GPU memory

# Feature flags
USE_TRAIN_AUDIO = True
USE_PSEUDO_LABELING = True

# Pseudo-labeling parameters
PSEUDO_PARAMS = {
    "threshold_high": 0.95,  # Probability threshold for positive pseudo-labels
    "threshold_low": 0.05,   # Probability threshold for negative pseudo-labels
    "max_per_class": 500,    # Max pseudo-labels per class
    "min_per_class": 10,     # Min samples before using pseudo-labels
}

# Hyperparameters
BEST_PROBE = {
    "pca_dim": 64,
    "min_pos": 8,
    "C": 0.50,
    "alpha": 0.40,
    # MLP parameters
    "mlp_hidden": (128,),
    "mlp_activation": "relu",
    "mlp_max_iter": 300,
    "mlp_early_stopping": True,
    "mlp_validation_fraction": 0.15,
    "mlp_n_iter_no_change": 15,
    "mlp_learning_rate_init": 0.001,
    "mlp_alpha": 0.01,
    "mlp_random_state": 42,
}

# Ensemble settings (2024 1st place: min reduction)
BEST_ENSEMBLE = {
    "method": "min",
    "lr_weight": 0.5,
    "mlp_weight": 0.5,
}

BEST_FUSION = {
    "lambda_event": 0.4,
    "lambda_texture": 1.0,
    "lambda_proxy_texture": 0.8,
    "smooth_texture": 0.35,
}

print("=" * 70)
print("BirdCLEF+ 2026 - Training v3 (GPU + Advanced Features)")
print("=" * 70)
print(f"USE_TRAIN_AUDIO: {USE_TRAIN_AUDIO}")
print(f"USE_PSEUDO_LABELING: {USE_PSEUDO_LABELING}")

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

# Handle duplicates (discovered by @ttahara)
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
    """Parse semicolon-separated labels."""
    if pd.isna(x):
        return []
    return [t.strip() for t in str(x).split(";") if t.strip()]


def union_labels(series):
    """Union of all labels in a series."""
    return sorted(set(lbl for x in series for lbl in parse_labels(x)))


FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")


def parse_filename(name):
    """Extract site and hour from soundscape filename."""
    m = FNAME_RE.match(name)
    if not m:
        return {"site": None, "hour_utc": -1}
    _, site, _, hms = m.groups()
    return {"site": site, "hour_utc": int(hms[:2])}


# Process soundscape labels
sc_clean = soundscape_labels.groupby(["filename", "start", "end"])["primary_label"].apply(union_labels).reset_index(name="label_list")
sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
sc_clean["row_id"] = sc_clean["filename"].str.replace(".ogg", "", regex=False) + "_" + sc_clean["end_sec"].astype(str)

meta = sc_clean["filename"].apply(parse_filename).apply(pd.Series)
sc_clean = pd.concat([sc_clean, meta], axis=1)

# Get all labeled files
all_labeled_files = sorted(sc_clean["filename"].unique().tolist())

# Get all soundscape files (including unlabeled)
all_soundscape_files = sorted([f.name for f in (BASE / "train_soundscapes").glob("*.ogg")])
unlabeled_files = sorted(set(all_soundscape_files) - set(all_labeled_files))

print(f"Labeled soundscape files: {len(all_labeled_files)}")
print(f"Unlabeled soundscape files: {len(unlabeled_files)}")

# Build label matrix for labeled soundscapes
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

# Load Perch labels
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

# Class types (texture = insects/amphibians, event = birds/mammals)
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
# FROG PROXIES (for unmapped amphibians)
# =============================================================================
print("\nBuilding frog proxies for unmapped amphibians...")

unmapped_df = mapping[mapping["bc_index"] == NO_LABEL_INDEX].copy()
unmapped_non_sonotype = unmapped_df[~unmapped_df["primary_label"].astype(str).str.contains("son", na=False)].copy()


def get_genus_hits(sci_name):
    """Find Perch labels matching the genus."""
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
    """Calculate macro-averaged ROC-AUC, skipping classes with no positives."""
    keep = y_true.sum(axis=0) > 0
    if keep.sum() == 0:
        return 0.0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")


def sigmoid(x):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def smooth_cols(scores, cols, alpha=0.35):
    """Temporal smoothing for texture classes."""
    if alpha <= 0 or len(cols) == 0:
        return scores.copy()
    s = scores.copy()
    n_samples = s.shape[0]
    if n_samples % N_WINDOWS != 0:
        return s  # Cannot reshape
    view = s.reshape(-1, N_WINDOWS, s.shape[1])
    x = view[:, :, cols]
    prev_x = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    next_x = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    view[:, :, cols] = (1.0 - alpha) * x + 0.5 * alpha * (prev_x + next_x)
    return s


def seq_features_1d(v):
    """Extract sequence features (prev, next, mean, max)."""
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
    """Build feature vector for probe training."""
    prev_b, next_b, mean_b, max_b = seq_features_1d(base)
    return np.concatenate([
        emb,
        raw[:, None],
        prior[:, None],
        base[:, None],
        prev_b[:, None],
        next_b[:, None],
        mean_b[:, None],
        max_b[:, None]
    ], axis=1).astype(np.float32)


def ensemble_predictions(lr_scores, mlp_scores, method="min", lr_weight=0.5, mlp_weight=0.5):
    """Ensemble LR and MLP predictions (2024 1st place: min reduction)."""
    if method == "min":
        return np.minimum(lr_scores * lr_weight, mlp_scores * mlp_weight) * 2
    elif method == "avg":
        return (lr_scores + mlp_scores) / 2
    else:
        return lr_scores * lr_weight + mlp_scores * mlp_weight

# =============================================================================
# PRIOR TABLES
# =============================================================================
def fit_prior_tables(df, Y):
    """Build prior probability tables from labeled data."""
    df = df.reset_index(drop=True)
    global_p = Y.mean(axis=0).astype(np.float32)

    # Site priors
    site_keys = sorted(df["site"].dropna().astype(str).unique().tolist())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_n = np.zeros(len(site_keys), dtype=np.float32)
    site_p = np.zeros((len(site_keys), Y.shape[1]), dtype=np.float32)
    for s in site_keys:
        m = df["site"].astype(str).values == s
        site_n[site_to_i[s]] = m.sum()
        site_p[site_to_i[s]] = Y[m].mean(axis=0)

    # Hour priors
    hour_keys = sorted(df["hour_utc"].dropna().astype(int).unique().tolist())
    hour_to_i = {h: i for i, h in enumerate(hour_keys)}
    hour_n = np.zeros(len(hour_keys), dtype=np.float32)
    hour_p = np.zeros((len(hour_keys), Y.shape[1]), dtype=np.float32)
    for h in hour_keys:
        m = df["hour_utc"].astype(int).values == h
        hour_n[hour_to_i[h]] = m.sum()
        hour_p[hour_to_i[h]] = Y[m].mean(axis=0)

    # Site-hour joint priors
    sh_to_i = {}
    sh_n_list, sh_p_list = [], []
    for (s, h), idx in df.groupby(["site", "hour_utc"]).groups.items():
        sh_to_i[(str(s), int(h))] = len(sh_n_list)
        idx = np.array(list(idx))
        sh_n_list.append(len(idx))
        sh_p_list.append(Y[idx].mean(axis=0))

    return {
        "global_p": global_p,
        "site_to_i": site_to_i,
        "site_n": site_n,
        "site_p": site_p,
        "hour_to_i": hour_to_i,
        "hour_n": hour_n,
        "hour_p": hour_p,
        "sh_to_i": sh_to_i,
        "sh_n": np.array(sh_n_list, dtype=np.float32),
        "sh_p": np.stack(sh_p_list).astype(np.float32) if sh_p_list else np.zeros((0, Y.shape[1]), dtype=np.float32),
    }


def prior_logits(sites, hours, tables, eps=1e-4):
    """Calculate prior logits for given sites and hours."""
    n = len(sites)
    p = np.repeat(tables["global_p"][None, :], n, axis=0).astype(np.float32, copy=True)

    site_idx = np.fromiter((tables["site_to_i"].get(str(s), -1) for s in sites), dtype=np.int32, count=n)
    hour_idx = np.fromiter((tables["hour_to_i"].get(int(h), -1) if int(h) >= 0 else -1 for h in hours), dtype=np.int32, count=n)
    sh_idx = np.fromiter((tables["sh_to_i"].get((str(s), int(h)), -1) if int(h) >= 0 else -1 for s, h in zip(sites, hours)), dtype=np.int32, count=n)

    # Hour smoothing
    valid = hour_idx >= 0
    if valid.any():
        nh = tables["hour_n"][hour_idx[valid]][:, None]
        p[valid] = nh / (nh + 8.0) * tables["hour_p"][hour_idx[valid]] + (1.0 - nh / (nh + 8.0)) * p[valid]

    # Site smoothing
    valid = site_idx >= 0
    if valid.any():
        ns = tables["site_n"][site_idx[valid]][:, None]
        p[valid] = ns / (ns + 8.0) * tables["site_p"][site_idx[valid]] + (1.0 - ns / (ns + 8.0)) * p[valid]

    # Site-hour smoothing
    valid = sh_idx >= 0
    if valid.any():
        nsh = tables["sh_n"][sh_idx[valid]][:, None]
        p[valid] = nsh / (nsh + 4.0) * tables["sh_p"][sh_idx[valid]] + (1.0 - nsh / (nsh + 4.0)) * p[valid]

    np.clip(p, eps, 1.0 - eps, out=p)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


def fuse_scores(base, sites, hours, tables):
    """Fuse raw scores with prior information."""
    scores = base.copy()
    prior = prior_logits(sites, hours, tables)

    # Event classes (birds, mammals)
    if len(idx_mapped_active_event):
        scores[:, idx_mapped_active_event] += BEST_FUSION["lambda_event"] * prior[:, idx_mapped_active_event]

    # Texture classes (insects, amphibians)
    if len(idx_mapped_active_texture):
        scores[:, idx_mapped_active_texture] += BEST_FUSION["lambda_texture"] * prior[:, idx_mapped_active_texture]

    # Proxy-mapped texture classes
    if len(idx_selected_proxy_active_texture):
        scores[:, idx_selected_proxy_active_texture] += BEST_FUSION["lambda_proxy_texture"] * prior[:, idx_selected_proxy_active_texture]

    # Prior-only classes (unmapped)
    if len(idx_selected_prioronly_active_event):
        scores[:, idx_selected_prioronly_active_event] = BEST_FUSION["lambda_event"] * prior[:, idx_selected_prioronly_active_event]
    if len(idx_selected_prioronly_active_texture):
        scores[:, idx_selected_prioronly_active_texture] = BEST_FUSION["lambda_texture"] * prior[:, idx_selected_prioronly_active_texture]

    # Inactive unmapped classes
    if len(idx_unmapped_inactive):
        scores[:, idx_unmapped_inactive] = -8.0

    # Temporal smoothing for texture classes
    scores = smooth_cols(scores, idx_active_texture, alpha=BEST_FUSION["smooth_texture"])
    return scores.astype(np.float32), prior

# =============================================================================
# AUDIO PROCESSING
# =============================================================================
def read_audio(path, target_samples=FILE_SAMPLES):
    """Read and pad/trim audio to target length."""
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    elif len(y) > target_samples:
        y = y[:target_samples]
    return y


def read_soundscape_batch(paths):
    """Read multiple soundscape files in parallel."""
    def read_one(path):
        try:
            y = read_audio(path)
            return y
        except:
            return np.zeros(FILE_SAMPLES, dtype=np.float32)
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(read_one, paths))
    return np.stack(results, axis=0).astype(np.float32)

# =============================================================================
# PERCH INFERENCE - SOUNDSCAPES (OPTIMIZED)
# =============================================================================
def infer_perch_soundscapes(paths, verbose=True):
    """Run Perch inference on soundscape files (1-minute each).
    
    Optimized with parallel file loading for faster I/O.
    """
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

        # PARALLEL file loading (much faster than sequential)
        audio_batch = read_soundscape_batch(batch_paths)
        x = audio_batch.reshape(batch_size * N_WINDOWS, WINDOW_SAMPLES)

        # Extract metadata
        for path in batch_paths:
            meta = parse_filename(path.name)
            row_ids[write_row:write_row + N_WINDOWS] = [f"{path.stem}_{t}" for t in range(5, 65, 5)]
            sites[write_row:write_row + N_WINDOWS] = meta["site"]
            hours[write_row:write_row + N_WINDOWS] = int(meta["hour_utc"])
            write_row += N_WINDOWS

        # GPU inference with memory management
        with tf.device('/GPU:0'):
            outputs = infer_fn(inputs=tf.convert_to_tensor(x))
            logits = outputs["label"].numpy().astype(np.float32)
            emb = outputs["embedding"].numpy().astype(np.float32)

        # Map scores
        scores[batch_start:write_row, MAPPED_POS] = logits[:write_row - batch_start, MAPPED_BC_INDICES]
        embeddings[batch_start:write_row] = emb

        # Proxy scores
        for pos, bc_idx_arr in selected_proxy_pos_to_bc.items():
            scores[batch_start:write_row, pos] = logits[:write_row - batch_start, bc_idx_arr].max(axis=1)

        # Clear GPU memory
        del x, outputs, logits, emb
        tf.keras.backend.clear_session()
        gc.collect()

    return pd.DataFrame({"row_id": row_ids, "site": sites, "hour_utc": hours}), scores, embeddings


# =============================================================================
# PARALLEL AUDIO LOADING (Speed Optimization)
# =============================================================================
def load_single_audio(path):
    """Load and process a single audio file."""
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if len(y) < WINDOW_SAMPLES:
            y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
        elif len(y) > WINDOW_SAMPLES:
            # Take center of audio
            center = (len(y) - WINDOW_SAMPLES) // 2
            y = y[center:center + WINDOW_SAMPLES]
        return y
    except Exception as e:
        return np.zeros(WINDOW_SAMPLES, dtype=np.float32)


def load_audio_batch_parallel(paths, n_workers=8):
    """Load audio files in parallel using ThreadPoolExecutor.
    
    This provides 4-8x speedup over sequential loading by using
    multiple CPU threads for disk I/O while GPU processes.
    """
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(load_single_audio, paths))
    return np.stack(results, axis=0).astype(np.float32)


# =============================================================================
# PERCH INFERENCE - TRAIN_AUDIO (SHORT CLIPS) - OPTIMIZED
# =============================================================================
def infer_perch_short(paths, labels, verbose=True):
    """Infer Perch on short train_audio clips (single window per clip).
    
    Optimized with parallel file loading for 4-8x speedup.
    """
    paths = [Path(p) for p in paths]
    n_files = len(paths)

    scores = np.zeros((n_files, N_CLASSES), dtype=np.float32)
    embeddings = np.zeros((n_files, 1536), dtype=np.float32)

    # More files per batch since 1 window each (48 files per batch)
    batch_size = BATCH_FILES * N_WINDOWS
    iterator = range(0, n_files, batch_size)
    if verbose:
        iterator = tqdm(iterator, total=(n_files + batch_size - 1) // batch_size, desc="Perch Short")

    for start in iterator:
        batch_paths = paths[start:start + batch_size]
        n_batch = len(batch_paths)

        # PARALLEL file loading (4-8x faster than sequential)
        x = load_audio_batch_parallel(batch_paths, n_workers=8)

        # GPU inference
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

    # Build label matrix
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

# Filter to only windows that have labels
labeled_row_ids = set(sc_clean["row_id"].tolist())
keep_mask = meta_all["row_id"].isin(labeled_row_ids).values

meta_all = meta_all[keep_mask].reset_index(drop=True)
scores_raw = scores_raw[keep_mask]
emb_all = emb_all[keep_mask]

# Build Y_TRAIN by looking up labels
row_id_to_label_idx = {row_id: idx for idx, row_id in enumerate(sc_clean["row_id"])}
Y_TRAIN = np.zeros((len(meta_all), N_CLASSES), dtype=np.uint8)
for i, row_id in enumerate(meta_all["row_id"]):
    Y_TRAIN[i] = Y_SC[row_id_to_label_idx[row_id]]

print(f"Total windows processed: {len(meta_all)} (filtered to labeled windows only)")
print(f"scores_raw: {scores_raw.shape}, emb_all: {emb_all.shape}")
print(f"Y_TRAIN: {Y_TRAIN.shape}, positive rate: {Y_TRAIN.mean():.4%}")

# =============================================================================
# RUN PERCH ON TRAIN_AUDIO (SHORT CLIPS) - OPTIONAL
# =============================================================================
if USE_TRAIN_AUDIO:
    print("\n" + "-" * 70)
    print("Running Perch on train_audio (short clips)...")
    print("-" * 70)

    # Filter to valid files with labels in competition
    valid_audio = train_audio_meta[train_audio_meta["primary_label"].isin(PRIMARY_LABELS)].copy()
    print(f"Valid train_audio files: {len(valid_audio)}")

    # Build paths and labels
    audio_paths = [BASE / "train_audio" / row["filename"] for _, row in valid_audio.iterrows()]
    audio_labels = valid_audio["primary_label"].tolist()

    # Process in chunks to avoid memory issues
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
    print(f"scores_audio: {scores_audio.shape}, emb_audio: {emb_audio.shape}")
    print(f"Species with train_audio data: {(Y_audio.sum(axis=0) > 0).sum()}")
else:
    scores_audio = np.zeros((0, N_CLASSES), dtype=np.float32)
    emb_audio = np.zeros((0, 1536), dtype=np.float32)
    Y_audio = np.zeros((0, N_CLASSES), dtype=np.uint8)
    print("\nSkipping train_audio (USE_TRAIN_AUDIO=False)")

# =============================================================================
# COMBINE SOUNDSCAPE + TRAIN_AUDIO DATA
# =============================================================================
print("\n" + "-" * 70)
print("Combining soundscape + train_audio data...")
print("-" * 70)

emb_combined = np.concatenate([emb_all, emb_audio], axis=0)
scores_combined = np.concatenate([scores_raw, scores_audio], axis=0)
Y_COMBINED = np.concatenate([Y_TRAIN, Y_audio], axis=0)

print(f"Combined: {len(emb_combined)} samples")
print(f"  - Soundscapes: {len(emb_all)}")
print(f"  - Train audio: {len(emb_audio)}")
print(f"Species with data: {(Y_COMBINED.sum(axis=0) > 0).sum()}")

# =============================================================================
# BUILD OOF (Soundscapes only - for validation)
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

# For train_audio, use raw scores (no prior fusion - no site/hour info)
audio_prior = np.zeros_like(scores_audio, dtype=np.float32) if len(scores_audio) > 0 else np.zeros((0, N_CLASSES), dtype=np.float32)
audio_base = scores_audio.copy() if len(scores_audio) > 0 else np.zeros((0, N_CLASSES), dtype=np.float32)

# =============================================================================
# TRAIN INITIAL PROBES (LR + MLP)
# =============================================================================
print("\n" + "-" * 70)
print("Training initial probes...")
print("-" * 70)

emb_scaler = StandardScaler()
emb_scaled = emb_scaler.fit_transform(emb_combined)

n_comp = min(int(BEST_PROBE["pca_dim"]), emb_scaled.shape[0] - 1, emb_scaled.shape[1])
emb_pca = PCA(n_components=n_comp)
Z_combined = emb_pca.fit_transform(emb_scaled).astype(np.float32)

print(f"PCA: {n_comp}, var: {emb_pca.explained_variance_ratio_.sum():.4f}")

pos_counts = Y_COMBINED.sum(axis=0)
probe_idx = np.where(pos_counts >= int(BEST_PROBE["min_pos"]))[0].astype(np.int32)

# Combine prior/base for training
prior_combined = np.concatenate([oof_prior, audio_prior], axis=0) if len(audio_prior) > 0 else oof_prior
base_combined = np.concatenate([oof_base, audio_base], axis=0) if len(audio_base) > 0 else oof_base

# Train LR probes
lr_probes = {}
for cls_idx in tqdm(probe_idx, desc="LR Probes"):
    y = Y_COMBINED[:, cls_idx]
    if y.sum() == 0 or y.sum() == len(y):
        continue
    X = build_class_features(Z_combined, scores_combined[:, cls_idx], prior_combined[:, cls_idx], base_combined[:, cls_idx])
    clf = LogisticRegression(C=float(BEST_PROBE["C"]), max_iter=400, solver="liblinear", class_weight="balanced")
    clf.fit(X, y)
    lr_probes[cls_idx] = clf

# Train MLP probes
mlp_probes = {}
for cls_idx in tqdm(probe_idx, desc="MLP Probes"):
    y = Y_COMBINED[:, cls_idx]
    if y.sum() == 0 or y.sum() == len(y):
        continue
    X = build_class_features(Z_combined, scores_combined[:, cls_idx], prior_combined[:, cls_idx], base_combined[:, cls_idx])
    mlp = MLPClassifier(
        hidden_layer_sizes=BEST_PROBE["mlp_hidden"],
        activation=BEST_PROBE["mlp_activation"],
        max_iter=BEST_PROBE["mlp_max_iter"],
        early_stopping=BEST_PROBE["mlp_early_stopping"],
        validation_fraction=BEST_PROBE["mlp_validation_fraction"],
        n_iter_no_change=BEST_PROBE["mlp_n_iter_no_change"],
        learning_rate_init=BEST_PROBE["mlp_learning_rate_init"],
        alpha=BEST_PROBE["mlp_alpha"],
        random_state=BEST_PROBE["mlp_random_state"],
    )
    mlp.fit(X, y)
    mlp_probes[cls_idx] = mlp

print(f"\nTrained {len(lr_probes)} LR probes, {len(mlp_probes)} MLP probes")

# =============================================================================
# PSEUDO-LABELING
# =============================================================================
if USE_PSEUDO_LABELING and len(unlabeled_files) > 0:
    print("\n" + "=" * 70)
    print("PSEUDO-LABELING UNLABELED SOUNDSCAPES")
    print("=" * 70)

    # Run Perch on unlabeled soundscapes (limit for efficiency)
    print(f"\nProcessing {len(unlabeled_files)} unlabeled soundscape files...")
    unlabeled_paths = [BASE / "train_soundscapes" / fn for fn in unlabeled_files]
    meta_unlabeled, scores_unlabeled, emb_unlabeled = infer_perch_soundscapes(unlabeled_paths)

    print(f"Unlabeled embeddings: {emb_unlabeled.shape}")

    # Transform embeddings
    Z_unlabeled = emb_pca.transform(emb_scaler.transform(emb_unlabeled)).astype(np.float32)

    # Fuse with priors
    full_prior_tables = fit_prior_tables(sc_clean.reset_index(drop=True), Y_SC)
    unlabeled_base, unlabeled_prior = fuse_scores(
        scores_unlabeled,
        meta_unlabeled["site"].to_numpy(),
        meta_unlabeled["hour_utc"].to_numpy(),
        full_prior_tables
    )

    # Get predictions from both LR and MLP
    print("\nGenerating pseudo-labels...")

    # LR predictions
    lr_pred = unlabeled_base.copy()
    for cls_idx, clf in tqdm(lr_probes.items(), desc="LR Predictions"):
        X = build_class_features(Z_unlabeled, scores_unlabeled[:, cls_idx], unlabeled_prior[:, cls_idx], unlabeled_base[:, cls_idx])
        pred = clf.decision_function(X).astype(np.float32)
        lr_pred[:, cls_idx] = (1.0 - BEST_PROBE["alpha"]) * unlabeled_base[:, cls_idx] + BEST_PROBE["alpha"] * pred

    # MLP predictions
    mlp_pred = unlabeled_base.copy()
    for cls_idx, mlp in tqdm(mlp_probes.items(), desc="MLP Predictions"):
        X = build_class_features(Z_unlabeled, scores_unlabeled[:, cls_idx], unlabeled_prior[:, cls_idx], unlabeled_base[:, cls_idx])
        pred = mlp.predict_proba(X)[:, 1].astype(np.float32)
        pred_logit = np.log(np.clip(pred, 1e-7, 1 - 1e-7)) - np.log1p(-np.clip(pred, 1e-7, 1 - 1e-7))
        mlp_pred[:, cls_idx] = (1.0 - BEST_PROBE["alpha"]) * unlabeled_base[:, cls_idx] + BEST_PROBE["alpha"] * pred_logit

    # Ensemble predictions
    pseudo_scores = ensemble_predictions(lr_pred, mlp_pred, method=BEST_ENSEMBLE["method"],
                                         lr_weight=BEST_ENSEMBLE["lr_weight"], mlp_weight=BEST_ENSEMBLE["mlp_weight"])
    pseudo_probs = sigmoid(pseudo_scores)

    # Generate pseudo-labels with confidence threshold
    Y_pseudo = np.zeros((len(pseudo_probs), N_CLASSES), dtype=np.uint8)
    pseudo_counts = {}

    for cls_idx in probe_idx:
        if cls_idx not in lr_probes or cls_idx not in mlp_probes:
            continue

        probs = pseudo_probs[:, cls_idx]

        # Positive pseudo-labels (high confidence)
        pos_mask = probs >= PSEUDO_PARAMS["threshold_high"]
        pos_count = pos_mask.sum()

        # Limit per class
        if pos_count > PSEUDO_PARAMS["max_per_class"]:
            top_k_idx = np.argsort(probs)[-PSEUDO_PARAMS["max_per_class"]:]
            pos_mask = np.zeros(len(probs), dtype=bool)
            pos_mask[top_k_idx] = True

        Y_pseudo[pos_mask, cls_idx] = 1
        pseudo_counts[cls_idx] = pos_mask.sum()

    # Filter to only rows with at least one pseudo-label
    has_label = Y_pseudo.sum(axis=1) > 0
    Y_pseudo_filtered = Y_pseudo[has_label]
    emb_pseudo = emb_unlabeled[has_label]
    scores_pseudo = scores_unlabeled[has_label]
    prior_pseudo = unlabeled_prior[has_label]
    base_pseudo = unlabeled_base[has_label]

    print(f"\nPseudo-labels generated: {has_label.sum()} windows")
    print(f"Total pseudo-label count: {Y_pseudo_filtered.sum()}")

    # Combine with original data
    emb_combined_v2 = np.concatenate([emb_combined, emb_pseudo], axis=0)
    scores_combined_v2 = np.concatenate([scores_combined, scores_pseudo], axis=0)
    Y_COMBINED_V2 = np.concatenate([Y_COMBINED, Y_pseudo_filtered], axis=0)
    prior_combined_v2 = np.concatenate([prior_combined, prior_pseudo], axis=0)
    base_combined_v2 = np.concatenate([base_combined, base_pseudo], axis=0)

    print(f"\nCombined with pseudo-labels: {len(emb_combined_v2)} samples")

    # Retrain probes with pseudo-labels
    print("\n" + "-" * 70)
    print("Retraining probes with pseudo-labels...")
    print("-" * 70)

    # Re-fit scaler and PCA
    emb_scaler_v2 = StandardScaler()
    emb_scaled_v2 = emb_scaler_v2.fit_transform(emb_combined_v2)

    n_comp_v2 = min(int(BEST_PROBE["pca_dim"]), emb_scaled_v2.shape[0] - 1, emb_scaled_v2.shape[1])
    emb_pca_v2 = PCA(n_components=n_comp_v2)
    Z_combined_v2 = emb_pca_v2.fit_transform(emb_scaled_v2).astype(np.float32)

    # Retrain LR probes
    lr_probes_v2 = {}
    for cls_idx in tqdm(probe_idx, desc="LR Probes v2"):
        y = Y_COMBINED_V2[:, cls_idx]
        if y.sum() < PSEUDO_PARAMS["min_per_class"]:
            if cls_idx in lr_probes:
                lr_probes_v2[cls_idx] = lr_probes[cls_idx]
            continue
        X = build_class_features(Z_combined_v2, scores_combined_v2[:, cls_idx], prior_combined_v2[:, cls_idx], base_combined_v2[:, cls_idx])
        clf = LogisticRegression(C=float(BEST_PROBE["C"]), max_iter=400, solver="liblinear", class_weight="balanced")
        clf.fit(X, y)
        lr_probes_v2[cls_idx] = clf

    # Retrain MLP probes
    mlp_probes_v2 = {}
    for cls_idx in tqdm(probe_idx, desc="MLP Probes v2"):
        y = Y_COMBINED_V2[:, cls_idx]
        if y.sum() < PSEUDO_PARAMS["min_per_class"]:
            if cls_idx in mlp_probes:
                mlp_probes_v2[cls_idx] = mlp_probes[cls_idx]
            continue
        X = build_class_features(Z_combined_v2, scores_combined_v2[:, cls_idx], prior_combined_v2[:, cls_idx], base_combined_v2[:, cls_idx])
        mlp = MLPClassifier(
            hidden_layer_sizes=BEST_PROBE["mlp_hidden"],
            activation=BEST_PROBE["mlp_activation"],
            max_iter=BEST_PROBE["mlp_max_iter"],
            early_stopping=BEST_PROBE["mlp_early_stopping"],
            validation_fraction=BEST_PROBE["mlp_validation_fraction"],
            n_iter_no_change=BEST_PROBE["mlp_n_iter_no_change"],
            learning_rate_init=BEST_PROBE["mlp_learning_rate_init"],
            alpha=BEST_PROBE["mlp_alpha"],
            random_state=BEST_PROBE["mlp_random_state"],
        )
        mlp.fit(X, y)
        mlp_probes_v2[cls_idx] = mlp

    print(f"\nRetrained {len(lr_probes_v2)} LR probes, {len(mlp_probes_v2)} MLP probes")

    # Use v2 models for final
    emb_scaler_final = emb_scaler_v2
    emb_pca_final = emb_pca_v2
    lr_probes_final = lr_probes_v2
    mlp_probes_final = mlp_probes_v2

else:
    print("\nSkipping pseudo-labeling (USE_PSEUDO_LABELING=False or no unlabeled files)")
    emb_scaler_final = emb_scaler
    emb_pca_final = emb_pca
    lr_probes_final = lr_probes
    mlp_probes_final = mlp_probes
    has_label = np.zeros(0, dtype=bool)

# =============================================================================
# EVALUATE OOF WITH PROBES (Soundscapes only for validation)
# =============================================================================
print("\n" + "-" * 70)
print("Evaluating OOF with probes...")
print("-" * 70)

# Get PCA embeddings for soundscape data
Z_sc = emb_pca_final.transform(emb_scaler_final.transform(emb_all)).astype(np.float32)

# Get LR predictions
oof_lr = oof_base.copy()
for cls_idx, clf in lr_probes_final.items():
    if cls_idx < N_CLASSES:
        X = build_class_features(Z_sc, scores_raw[:, cls_idx], oof_prior[:, cls_idx], oof_base[:, cls_idx])
        pred = clf.decision_function(X).astype(np.float32)
        oof_lr[:, cls_idx] = (1.0 - BEST_PROBE["alpha"]) * oof_base[:, cls_idx] + BEST_PROBE["alpha"] * pred

# Get MLP predictions
oof_mlp = oof_base.copy()
for cls_idx, mlp in mlp_probes_final.items():
    if cls_idx < N_CLASSES:
        X = build_class_features(Z_sc, scores_raw[:, cls_idx], oof_prior[:, cls_idx], oof_base[:, cls_idx])
        pred = mlp.predict_proba(X)[:, 1].astype(np.float32)
        pred_logit = np.log(np.clip(pred, 1e-7, 1 - 1e-7)) - np.log1p(-np.clip(pred, 1e-7, 1 - 1e-7))
        oof_mlp[:, cls_idx] = (1.0 - BEST_PROBE["alpha"]) * oof_base[:, cls_idx] + BEST_PROBE["alpha"] * pred_logit

# Ensemble
oof_final = ensemble_predictions(
    oof_lr, oof_mlp,
    method=BEST_ENSEMBLE["method"],
    lr_weight=BEST_ENSEMBLE["lr_weight"],
    mlp_weight=BEST_ENSEMBLE["mlp_weight"]
)

print(f"\n--- Soundscape OOF Results ---")
print(f"OOF AUC (base):  {macro_auc(Y_TRAIN, oof_base):.6f}")
print(f"OOF AUC (LR):    {macro_auc(Y_TRAIN, oof_lr):.6f}")
print(f"OOF AUC (MLP):   {macro_auc(Y_TRAIN, oof_mlp):.6f}")
print(f"OOF AUC (final): {macro_auc(Y_TRAIN, oof_final):.6f}")

# =============================================================================
# SAVE ARTIFACTS
# =============================================================================
print("\n" + "-" * 70)
print("Saving artifacts...")
print("-" * 70)


def serialize_pickle(obj):
    """Serialize object to pickle bytes."""
    buf = BytesIO()
    pickle.dump(obj, buf)
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


# Save all arrays to single npz file
np.savez_compressed(
    OUTPUT_DIR / "full_perch_arrays.npz",
    # Soundscapes embeddings and scores
    embeddings=emb_all.astype(np.float32),
    scores_raw=scores_raw.astype(np.float32),
    oof_base=oof_base.astype(np.float32),
    oof_prior=oof_prior.astype(np.float32),
    oof_lr=oof_lr.astype(np.float32),
    oof_mlp=oof_mlp.astype(np.float32),
    # Train audio data
    emb_audio=emb_audio.astype(np.float32) if len(emb_audio) > 0 else np.zeros((0, 1536), dtype=np.float32),
    scores_audio=scores_audio.astype(np.float32) if len(scores_audio) > 0 else np.zeros((0, N_CLASSES), dtype=np.float32),
    # Index arrays
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
    # Config
    config=np.array(json.dumps({
        "best_probe": BEST_PROBE,
        "best_ensemble": BEST_ENSEMBLE,
        "best_fusion": BEST_FUSION,
        "pseudo_params": PSEUDO_PARAMS if USE_PSEUDO_LABELING else None,
        "n_classes": N_CLASSES,
        "n_windows": N_WINDOWS,
        "n_soundscapes": len(emb_all),
        "n_train_audio": len(emb_audio),
        "use_pseudo_labeling": USE_PSEUDO_LABELING,
        "use_train_audio": USE_TRAIN_AUDIO,
    })),
    labels=np.array(PRIMARY_LABELS, dtype=object),
    # Serialized models
    lr_probes_bytes=serialize_pickle(lr_probes_final),
    mlp_probes_bytes=serialize_pickle(mlp_probes_final),
    emb_scaler_bytes=serialize_pickle(emb_scaler_final),
    emb_pca_bytes=serialize_pickle(emb_pca_final),
    prior_tables_bytes=serialize_pickle(fit_prior_tables(sc_clean.reset_index(drop=True), Y_SC)),
)

# Save metadata
meta_all.to_parquet(OUTPUT_DIR / "full_perch_meta.parquet", index=False)

# Summary
print("\n" + "=" * 70)
print("Training Complete!")
print("=" * 70)
print(f"\nData Summary:")
print(f"  Soundscapes (labeled):  {len(emb_all)} windows")
print(f"  Train Audio:            {len(emb_audio)} clips")
if USE_PSEUDO_LABELING:
    print(f"  Pseudo-labeled:         {has_label.sum() if hasattr(has_label, 'sum') else 0} windows")
print(f"\nOutput Files:")
print(f"  - full_perch_arrays.npz    → embeddings + scores + models + indices")
print(f"  - full_perch_meta.parquet  → soundscape metadata")
print(f"\nMetrics:")
print(f"  Soundscape OOF AUC (final): {macro_auc(Y_TRAIN, oof_final):.6f}")