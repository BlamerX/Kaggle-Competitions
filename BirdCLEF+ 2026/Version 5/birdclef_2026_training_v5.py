"""
================================================================================
BirdCLEF+ 2026 - Training Script v5 
================================================================================

GPU VERSION - Run on Kaggle with GPU accelerator enabled

 (ALL computed during training):
  1. Pseudo-labeling - Uses unlabeled soundscapes for more training data
  2. Species Co-occurrence Features - Pre-computed interaction patterns
  3. Taxonomy-aware Grouping - Hierarchical species information

Output Files (same format as v2, compatible with v4 inference):
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
from collections import defaultdict
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
# CONFIG
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

# Hyperparameters
PCA_DIM = 128
MIN_POS = 4
LR_C = 0.7
BASE_ALPHA = 0.5

# Pseudo-labeling config
PSEUDO_THRESHOLD = 0.85  # Confidence threshold for pseudo-labels
MAX_PSEUDO_PER_CLASS = 50  # Max pseudo-labels per class

BEST_FUSION = {"lambda_event": 0.4, "lambda_texture": 1.0, "lambda_proxy_texture": 0.8, "smooth_texture": 0.35}

print("=" * 70)
print("BirdCLEF+ 2026 - Training v5 ")
print("=" * 70)
print(f"  1. Pseudo-labeling - threshold={PSEUDO_THRESHOLD}")
print(f"  2. Species Co-occurrence Features")
print(f"  3. Taxonomy-aware Grouping")

# =============================================================================
# LOAD DATA
# =============================================================================
print("\nLoading competition data...")

taxonomy = pd.read_csv(BASE / "taxonomy.csv")
sample_sub = pd.read_csv(BASE / "sample_submission.csv")
soundscape_labels = pd.read_csv(BASE / "train_soundscapes_labels.csv")

soundscape_labels = soundscape_labels.drop_duplicates().reset_index(drop=True)

PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES = len(PRIMARY_LABELS)
label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}

taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
soundscape_labels["primary_label"] = soundscape_labels["primary_label"].astype(str)

print(f"Classes: {N_CLASSES}, Raw label rows: {len(soundscape_labels)}")

# =============================================================================
# BUILD TAXONOMY GROUPINGS (IMPROVEMENT 3)
# =============================================================================
print("\n" + "-" * 70)
print("IMPROVEMENT 3: Building Taxonomy Groupings...")
print("-" * 70)

# Extract taxonomy hierarchy
taxonomy_info = {}
for _, row in taxonomy.iterrows():
    label = str(row["primary_label"])
    taxonomy_info[label] = {
        "class_name": row.get("class_name", "unknown"),
        "order": row.get("order", "unknown") if "order" in row else "unknown",
        "family": row.get("family", "unknown") if "family" in row else "unknown",
    }

# Build group indices
class_to_indices = defaultdict(list)
order_to_indices = defaultdict(list)
family_to_indices = defaultdict(list)

for i, label in enumerate(PRIMARY_LABELS):
    info = taxonomy_info.get(label, {})
    class_name = info.get("class_name", "unknown")
    order = info.get("order", "unknown")
    family = info.get("family", "unknown")
    
    class_to_indices[class_name].append(i)
    order_to_indices[order].append(i)
    family_to_indices[family].append(i)

# Convert to arrays for fast lookup
class_groups = {k: np.array(v, dtype=np.int32) for k, v in class_to_indices.items()}
order_groups = {k: np.array(v, dtype=np.int32) for k, v in order_to_indices.items()}
family_groups = {k: np.array(v, dtype=np.int32) for k, v in family_to_indices.items()}

# Map each class to its group members (for features)
class_to_class_group = {i: class_groups.get(taxonomy_info.get(label, {}).get("class_name", "unknown"), np.array([i])) for i, label in enumerate(PRIMARY_LABELS)}
class_to_order_group = {i: order_groups.get(taxonomy_info.get(label, {}).get("order", "unknown"), np.array([i])) for i, label in enumerate(PRIMARY_LABELS)}
class_to_family_group = {i: family_groups.get(taxonomy_info.get(label, {}).get("family", "unknown"), np.array([i])) for i, label in enumerate(PRIMARY_LABELS)}

print(f"Class groups: {len(class_groups)} ({list(class_groups.keys())[:5]}...)")
print(f"Order groups: {len(order_groups)}")
print(f"Family groups: {len(family_groups)}")

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
# BUILD CO-OCCURRENCE MATRIX (IMPROVEMENT 2)
# =============================================================================
print("\n" + "-" * 70)
print("IMPROVEMENT 2: Building Species Co-occurrence Matrix...")
print("-" * 70)

# Compute co-occurrence from training labels
co_occurrence = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float32)
for i in range(len(Y_SC)):
    labels = Y_SC[i]
    active = np.where(labels == 1)[0]
    for c1 in active:
        for c2 in active:
            co_occurrence[c1, c2] += 1

# Normalize to get conditional probabilities P(j|i)
row_sums = co_occurrence.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1  # Avoid division by zero
co_occurrence_prob = co_occurrence / row_sums

# Also compute pointwise mutual information (PMI)
# PMI(i,j) = log(P(i,j) / (P(i) * P(j)))
p_i = Y_SC.mean(axis=0)
p_ij = co_occurrence / len(Y_SC)

pmi_matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float32)
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        if p_i[i] > 0 and p_i[j] > 0 and p_ij[i, j] > 0:
            pmi_matrix[i, j] = np.log(p_ij[i, j] / (p_i[i] * p_i[j]))
        else:
            pmi_matrix[i, j] = 0

print(f"Co-occurrence matrix: {co_occurrence.shape}")
print(f"Top co-occurring pairs: {np.argsort(co_occurrence.sum(axis=1))[-5:]}")

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

def build_features_with_cooccurrence(emb, raw, prior, base, coocc_feat, taxonomy_feat):
    """Build features with co-occurrence and taxonomy features (pre-computed, no inference cost)."""
    prev_b, next_b, mean_b, max_b = seq_features_1d(base)
    return np.concatenate([
        emb,                                    # PCA embeddings
        raw[:, None],                          # Raw Perch score
        prior[:, None],                        # Prior logit
        base[:, None],                         # Base score
        prev_b[:, None],                       # Previous window
        next_b[:, None],                       # Next window
        mean_b[:, None],                       # File mean
        max_b[:, None],                        # File max
        coocc_feat,                            # Co-occurrence features (pre-computed)
        taxonomy_feat,                         # Taxonomy features (pre-computed)
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

# =============================================================================
# PRE-COMPUTE CO-OCCURRENCE FEATURES (IMPROVEMENT 2 - No inference cost)
# =============================================================================
print("\n" + "-" * 70)
print("IMPROVEMENT 2: Pre-computing Co-occurrence Features...")
print("-" * 70)

# For each sample, compute co-occurrence features based on predicted scores
# These are pre-computed during training, NOT during inference

def compute_cooccurrence_features(scores, cooccurrence_prob, pmi_matrix):
    """Compute co-occurrence features for all samples (pre-computed, stored)."""
    n_samples = len(scores)
    
    # Apply sigmoid to get probabilities
    probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -20, 20)))
    
    # Feature 1: Expected co-occurrence with each class
    expected_coocc = probs @ cooccurrence_prob
    
    # Feature 2: Max PMI with any class
    max_pmi = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        top_classes = np.argsort(probs[i])[-5:]  # Top 5 predicted classes
        max_pmi[i] = pmi_matrix[top_classes][:, top_classes].max()
    
    # Feature 3: Sum of probabilities for related classes (same class group)
    # This is computed per-class during feature building
    
    return expected_coocc, max_pmi

coocc_features, max_pmi_features = compute_cooccurrence_features(oof_base, co_occurrence_prob, pmi_matrix)
print(f"Co-occurrence features: {coocc_features.shape}, PMI features: {max_pmi_features.shape}")

# =============================================================================
# PRE-COMPUTE TAXONOMY FEATURES (IMPROVEMENT 3 - No inference cost)
# =============================================================================
print("\n" + "-" * 70)
print("IMPROVEMENT 3: Pre-computing Taxonomy Features...")
print("-" * 70)

def compute_taxonomy_features(scores, class_to_class_group, class_to_order_group):
    """Compute taxonomy-based features (pre-computed, stored)."""
    n_samples = len(scores)
    n_classes = scores.shape[1]
    
    # Apply sigmoid to get probabilities
    probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -20, 20)))
    
    # Feature 1: Max probability in same class group
    class_group_max = np.zeros((n_samples, n_classes), dtype=np.float32)
    for c in range(n_classes):
        group = class_to_class_group[c]
        class_group_max[:, c] = probs[:, group].max(axis=1)
    
    # Feature 2: Mean probability in same order
    order_group_mean = np.zeros((n_samples, n_classes), dtype=np.float32)
    for c in range(n_classes):
        group = class_to_order_group[c]
        if len(group) > 1:
            order_group_mean[:, c] = probs[:, group].mean(axis=1)
        else:
            order_group_mean[:, c] = probs[:, c]
    
    return class_group_max, order_group_mean

taxonomy_feat_class, taxonomy_feat_order = compute_taxonomy_features(oof_base, class_to_class_group, class_to_order_group)
print(f"Taxonomy features: {taxonomy_feat_class.shape}, {taxonomy_feat_order.shape}")

# =============================================================================
# TRAIN LR PROBES (Initial - for pseudo-labeling)
# =============================================================================
print("\n" + "-" * 70)
print("Training Initial LR Probes...")
print("-" * 70)

pos_counts = Y_TRAIN.sum(axis=0)
probe_idx = np.where(pos_counts >= MIN_POS)[0].astype(np.int32)
print(f"Classes with >= {MIN_POS} positives: {len(probe_idx)}")

# Build features with co-occurrence and taxonomy
def build_all_features(Z, scores_raw, oof_prior, oof_base, coocc_features, max_pmi_features, taxonomy_feat_class, taxonomy_feat_order, cls_idx):
    """Build features for a specific class including co-occurrence and taxonomy."""
    prev_b, next_b, mean_b, max_b = seq_features_1d(oof_base[:, cls_idx])
    
    # Base features
    base_feat = np.concatenate([
        Z,
        scores_raw[:, cls_idx][:, None],
        oof_prior[:, cls_idx][:, None],
        oof_base[:, cls_idx][:, None],
        prev_b[:, None],
        next_b[:, None],
        mean_b[:, None],
        max_b[:, None],
    ], axis=1)
    
    # Co-occurrence features
    coocc_feat = np.concatenate([
        coocc_features[:, cls_idx:cls_idx+1],
        max_pmi_features[:, None],
    ], axis=1)
    
    # Taxonomy features
    tax_feat = np.concatenate([
        taxonomy_feat_class[:, cls_idx:cls_idx+1],
        taxonomy_feat_order[:, cls_idx:cls_idx+1],
    ], axis=1)
    
    return np.concatenate([base_feat, coocc_feat, tax_feat], axis=1).astype(np.float32)

# Train initial probes
lr_probes = {}
for cls_idx in tqdm(probe_idx, desc="Initial LR"):
    y = Y_TRAIN[:, cls_idx]
    if y.sum() == 0 or y.sum() == len(y):
        continue
    X = build_all_features(Z, scores_raw, oof_prior, oof_base, coocc_features, max_pmi_features, taxonomy_feat_class, taxonomy_feat_order, cls_idx)
    clf = LogisticRegression(C=LR_C, max_iter=400, solver="liblinear", class_weight="balanced")
    clf.fit(X, y)
    lr_probes[cls_idx] = clf

print(f"Trained {len(lr_probes)} initial LR probes")

# =============================================================================
# PSEUDO-LABELING (IMPROVEMENT 1 - No inference cost)
# =============================================================================
print("\n" + "-" * 70)
print("IMPROVEMENT 1: Pseudo-Labeling...")
print("-" * 70)

# Get OOF predictions from initial model
oof_pred = oof_base.copy()
for cls_idx in tqdm(lr_probes.keys(), desc="OOF Predictions"):
    X = build_all_features(Z, scores_raw, oof_prior, oof_base, coocc_features, max_pmi_features, taxonomy_feat_class, taxonomy_feat_order, cls_idx)
    pred = lr_probes[cls_idx].decision_function(X)
    oof_pred[:, cls_idx] = (1.0 - BASE_ALPHA) * oof_base[:, cls_idx] + BASE_ALPHA * pred

# Apply sigmoid
oof_proba = 1.0 / (1.0 + np.exp(-np.clip(oof_pred, -20, 20)))

# Generate pseudo-labels
pseudo_labels = np.zeros_like(Y_TRAIN, dtype=np.uint8)
pseudo_weights = np.zeros(len(Y_TRAIN), dtype=np.float32)

for cls_idx in probe_idx:
    if cls_idx not in lr_probes:
        continue
    
    # Get high-confidence predictions
    proba = oof_proba[:, cls_idx]
    confident_mask = proba >= PSEUDO_THRESHOLD
    
    # Exclude already labeled samples
    already_labeled = Y_TRAIN[:, cls_idx] == 1
    pseudo_mask = confident_mask & (~already_labeled)
    
    # Limit per class
    pseudo_indices = np.where(pseudo_mask)[0]
    if len(pseudo_indices) > MAX_PSEUDO_PER_CLASS:
        # Keep highest confidence
        top_indices = np.argsort(proba[pseudo_mask])[-MAX_PSEUDO_PER_CLASS:]
        pseudo_indices = pseudo_indices[top_indices]
    
    if len(pseudo_indices) > 0:
        pseudo_labels[pseudo_indices, cls_idx] = 1
        pseudo_weights[pseudo_indices] = np.maximum(pseudo_weights[pseudo_indices], proba[pseudo_indices])

# Combine labels
Y_COMBINED = np.maximum(Y_TRAIN, pseudo_labels)
n_pseudo = (pseudo_labels.sum(axis=1) > 0).sum()
n_new_labels = pseudo_labels.sum()

print(f"Pseudo-labeled samples: {n_pseudo}")
print(f"Total pseudo-labels added: {n_new_labels}")
print(f"Combined label density: {Y_COMBINED.mean():.2%} (was {Y_TRAIN.mean():.2%})")

# =============================================================================
# RETRAIN WITH PSEUDO-LABELS
# =============================================================================
print("\n" + "-" * 70)
print("Retraining with Pseudo-labels...")
print("-" * 70)

lr_probes_final = {}
for cls_idx in tqdm(probe_idx, desc="Final LR"):
    y = Y_COMBINED[:, cls_idx]
    if y.sum() == 0 or y.sum() == len(y):
        continue
    X = build_all_features(Z, scores_raw, oof_prior, oof_base, coocc_features, max_pmi_features, taxonomy_feat_class, taxonomy_feat_order, cls_idx)
    clf = LogisticRegression(C=LR_C, max_iter=400, solver="liblinear", class_weight="balanced")
    clf.fit(X, y)
    lr_probes_final[cls_idx] = clf

print(f"Trained {len(lr_probes_final)} final LR probes")

# =============================================================================
# EVALUATE
# =============================================================================
print("\nEvaluating OOF...")

oof_final = oof_base.copy()
for cls_idx in tqdm(lr_probes_final.keys(), desc="Final Predictions"):
    X = build_all_features(Z, scores_raw, oof_prior, oof_base, coocc_features, max_pmi_features, taxonomy_feat_class, taxonomy_feat_order, cls_idx)
    pred = lr_probes_final[cls_idx].decision_function(X)
    oof_final[:, cls_idx] = (1.0 - BASE_ALPHA) * oof_base[:, cls_idx] + BASE_ALPHA * pred

print(f"OOF AUC (base):        {macro_auc(Y_TRAIN, oof_base):.6f}")
print(f"OOF AUC (with pseudo): {macro_auc(Y_TRAIN, oof_final):.6f}")

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
    
    if result.x >= 9.9 or result.x <= 0.1:
        temperatures = np.linspace(0.1, 20.0, 100)
        nlls = [nll(t) for t in temperatures]
        best_idx = np.argmin(nlls)
        return temperatures[best_idx]
    
    return result.x

all_logits = oof_final.flatten()
all_labels = Y_TRAIN.flatten()
sample_idx = np.random.choice(len(all_logits), min(100000, len(all_logits)), replace=False)
optimal_temp = find_optimal_temperature(all_logits[sample_idx], all_labels[sample_idx])

print(f"Optimal temperature: {optimal_temp:.4f}")

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
    # Pre-computed features 
    cooccurrence_prob=co_occurrence_prob,
    pmi_matrix=pmi_matrix,
    config=np.array(json.dumps({
        "pca_dim": PCA_DIM,
        "min_pos": MIN_POS,
        "lr_c": LR_C,
        "probe_alpha": BASE_ALPHA,
        "best_fusion": BEST_FUSION,
        "n_classes": N_CLASSES,
        "n_windows": N_WINDOWS,
        "pca_n_components": n_comp,
        "temperature": optimal_temp,
        "pseudo_threshold": PSEUDO_THRESHOLD,
        "use_pseudo_labeling": True,
        "use_cooccurrence": True,
        "use_taxonomy": True,
    })),
    labels=np.array(PRIMARY_LABELS, dtype=object),
    lr_probes_bytes=serialize_pickle(lr_probes_final),
    mlp_probes_bytes=serialize_pickle({}),  # Empty for v4 compatibility
    emb_scaler_bytes=serialize_pickle(emb_scaler),
    emb_pca_bytes=serialize_pickle(emb_pca),
    prior_tables_bytes=serialize_pickle(prior_tables),
)

meta_train.to_parquet(OUTPUT_DIR / "full_perch_meta.parquet", index=False)

print("\n" + "=" * 70)
print("Training Complete!")
print("=" * 70)
print(f"\nOutput Files:")
print(f"  - full_perch_arrays.npz")
print(f"  - full_perch_meta.parquet")
print(f"  1. Pseudo-labeling: {n_pseudo} samples, {n_new_labels} labels added")
print(f"  2. Co-occurrence Features: {coocc_features.shape[1]} features")
print(f"  3. Taxonomy Features: {taxonomy_feat_class.shape[1] + taxonomy_feat_order.shape[1]} features")
print(f"  4. Temperature: {optimal_temp:.4f}")
print(f"\nMetrics:")
print(f"  OOF AUC (base):        {macro_auc(Y_TRAIN, oof_base):.6f}")
print(f"  OOF AUC (improved):    {macro_auc(Y_TRAIN, oof_final):.6f}")