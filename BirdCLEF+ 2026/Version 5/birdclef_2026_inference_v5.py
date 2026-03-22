"""
================================================================================
BirdCLEF+ 2026 - Inference Script v5 (CPU Submission)
================================================================================

Uses pre-trained LR probes from v5 GPU training phase.
Runs on CPU within 90-minute Kaggle submission constraint.

v5 Improvements (computed during training, ZERO extra inference cost):
  1. Pseudo-labeling - Model already trained with extra data
  2. Temperature Scaling - Single scalar multiply

The training uses simple features (no co-occurrence/taxonomy matrices needed).

================================================================================
"""

# =============================================================================
# INSTALL TENSORFLOW 2.20 (Required for Perch v2 CPU)
# =============================================================================
!pip install -q --no-deps /kaggle/input/notebooks/kdmitrie/bc26-tensorflow-2-20-0/wheel/tensorboard-2.20.0-py3-none-any.whl
!pip install -q --no-deps /kaggle/input/notebooks/kdmitrie/bc26-tensorflow-2-20-0/wheel/tensorflow-2.20.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU for inference

import gc
import json
import pickle
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
tf.experimental.numpy.experimental_enable_numpy_behavior()

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE = Path("/kaggle/input/competitions/birdclef-2026")
MODELS_DIR = Path("/kaggle/input/notebooks/blamerx/birdclef-2026-training")
PERCH_CPU_PATH = "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1"
OUTPUT_DIR = Path("/kaggle/working")

# Audio parameters
SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12
BATCH_FILES = 50

# Training artifacts
ARRAYS_FILE = MODELS_DIR / "full_perch_arrays.npz"

print("=" * 70)
print("BirdCLEF+ 2026 - Inference v5 (CPU Submission)")
print("=" * 70)

# =============================================================================
# LOAD TRAINING ARTIFACTS
# =============================================================================
print("\n" + "-" * 70)
print("Loading training artifacts...")
print("-" * 70)

arrays = np.load(ARRAYS_FILE, allow_pickle=True)
print(f"Loaded arrays from: {ARRAYS_FILE}")
print(f"Keys in arrays: {list(arrays.keys())}")

# Load config
config = json.loads(str(arrays['config']))

N_CLASSES = config['n_classes']
N_WINDOWS = config['n_windows']
BEST_FUSION = config['best_fusion']
PROBE_ALPHA = config.get('probe_alpha', 0.5)
TEMPERATURE = config.get('temperature', 1.0)

print(f"\nConfig: PCA={config['pca_dim']}, LR_C={config['lr_c']}, Alpha={PROBE_ALPHA}")
print(f"Temperature: {TEMPERATURE:.4f}")

# Load labels
PRIMARY_LABELS = arrays['labels'].tolist()

# Load index arrays
BC_INDICES = arrays['bc_indices']
MAPPED_POS = arrays['mapped_pos']
MAPPED_BC_INDICES = arrays['mapped_bc_indices']
selected_proxy_pos = arrays['selected_proxy_pos']
idx_active_texture = arrays['idx_active_texture']
idx_active_event = arrays['idx_active_event']
idx_mapped_active_texture = arrays['idx_mapped_active_texture']
idx_mapped_active_event = arrays['idx_mapped_active_event']
idx_selected_proxy_active_texture = arrays['idx_selected_proxy_active_texture']
idx_selected_prioronly_active_texture = arrays['idx_selected_prioronly_active_texture']
idx_selected_prioronly_active_event = arrays['idx_selected_prioronly_active_event']
idx_unmapped_inactive = arrays['idx_unmapped_inactive']

# Proxy map
proxy_map_keys = arrays['proxy_map_keys']
proxy_map_vals = arrays['proxy_map_vals']
selected_proxy_pos_to_bc = {int(k): np.array(v, dtype=np.int32) for k, v in zip(proxy_map_keys, proxy_map_vals)}

# Check for co-occurrence matrices
cooccurrence_prob = arrays.get('cooccurrence_prob', None)
pmi_matrix = arrays.get('pmi_matrix', None)
has_cooccurrence = cooccurrence_prob is not None
print(f"Co-occurrence matrix: {cooccurrence_prob.shape if has_cooccurrence else 'Not found'}")
print(f"PMI matrix: {pmi_matrix.shape if pmi_matrix is not None else 'Not found'}")

# Deserialize models
def deserialize_pickle(arr):
    buf = BytesIO(arr.tobytes())
    return pickle.load(buf)

print("\nDeserializing models...")
lr_probes = deserialize_pickle(arrays['lr_probes_bytes'])
emb_scaler = deserialize_pickle(arrays['emb_scaler_bytes'])
emb_pca = deserialize_pickle(arrays['emb_pca_bytes'])
prior_tables = deserialize_pickle(arrays['prior_tables_bytes'])

print(f"LR probes: {len(lr_probes)}")

# Detect feature dimension from model
sample_clf = list(lr_probes.values())[0]
n_features_expected = sample_clf.n_features_in_
print(f"Model expects {n_features_expected} features")

# =============================================================================
# LOAD PERCH V2 CPU MODEL
# =============================================================================
print("\n" + "-" * 70)
print("Loading Perch v2 model (CPU)...")
print("-" * 70)

perch_model = tf.saved_model.load(PERCH_CPU_PATH)
infer_fn = perch_model.signatures["serving_default"]
print(f"Perch model loaded from: {PERCH_CPU_PATH}")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def sigmoid(x, temperature=1.0):
    """Numerically stable sigmoid with temperature scaling."""
    return 1.0 / (1.0 + np.exp(-np.clip(x / temperature, -20, 20)))


FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")


def parse_filename(name):
    m = FNAME_RE.match(name)
    if not m:
        return {"site": None, "hour_utc": -1}
    _, site, _, hms = m.groups()
    return {"site": site, "hour_utc": int(hms[:2])}


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
    """Compute sequence features for 1D array."""
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


def build_features(emb, raw, prior, base):
    """
    Build base features (same as training).
    
    Dimensions:
    - emb: (n_samples, pca_dim) = 128
    - raw, prior, base, prev, next, mean, max: 1 each
    Total: 128 + 7 = 135
    """
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


def build_features_with_coocc(emb, raw, prior, base, all_probs, cls_idx, coocc_prob, pmi):
    """
    Build features with co-occurrence and taxonomy features.
    
    Dimensions:
    - Base: 128 (PCA) + 7 (seq) = 135
    - Co-occurrence: expected_coocc (1) + max_pmi (1) = 2
    - Taxonomy: class_max (1) + order_mean (1) = 2
    Total: 135 + 4 = 139
    """
    # Base features
    prev_b, next_b, mean_b, max_b = seq_features_1d(base)
    base_feat = np.concatenate([
        emb,
        raw[:, None],
        prior[:, None],
        base[:, None],
        prev_b[:, None],
        next_b[:, None],
        mean_b[:, None],
        max_b[:, None]
    ], axis=1)
    
    # Co-occurrence feature 1: Expected co-occurrence for this class
    # P(j|i) weighted by current prediction distribution
    if coocc_prob is not None:
        expected_coocc_all = all_probs @ coocc_prob  # (n_samples, n_classes)
        expected_coocc = expected_coocc_all[:, cls_idx]  # (n_samples,)
    else:
        expected_coocc = np.zeros(len(base), dtype=np.float32)
    
    # Co-occurrence feature 2: Max PMI among top predicted classes
    if pmi is not None:
        top_k = 5
        top_classes = np.argsort(all_probs, axis=1)[:, -top_k:]  # (n_samples, top_k)
        max_pmi = np.zeros(len(base), dtype=np.float32)
        for i in range(len(base)):
            tc = top_classes[i]
            if len(tc) > 1:
                sub_pmi = pmi[tc][:, tc]
                max_pmi[i] = sub_pmi.max() if sub_pmi.size > 0 else 0.0
    else:
        max_pmi = np.zeros(len(base), dtype=np.float32)
    
    # Taxonomy feature 1: Max prob across all classes
    class_max = all_probs.max(axis=1)
    
    # Taxonomy feature 2: Mean prob across all classes
    order_mean = all_probs.mean(axis=1)
    
    # Combine: 135 + 4 = 139
    extra_feat = np.column_stack([expected_coocc, max_pmi, class_max, order_mean])
    
    return np.concatenate([base_feat, extra_feat], axis=1).astype(np.float32)


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


# =============================================================================
# PERCH INFERENCE (CPU)
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
        iterator = tqdm(iterator, total=(n_files + BATCH_FILES - 1) // BATCH_FILES, desc="Perch")

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
# MAIN INFERENCE
# =============================================================================
print("\n" + "-" * 70)
print("Running Inference")
print("-" * 70)

DRYRUN_N_FILES = 20
test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))

if len(test_paths) == 0:
    print(f"No test files. Using {DRYRUN_N_FILES} train soundscapes.")
    test_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:DRYRUN_N_FILES]
else:
    print(f"Test files: {len(test_paths)}")

# Run Perch
meta_test, scores_test, emb_test = infer_perch_soundscapes(test_paths)
print(f"scores_raw: {scores_test.shape}, emb_test: {emb_test.shape}")

# =============================================================================
# BUILD PREDICTIONS
# =============================================================================
print("\nBuilding predictions...")

# Transform embeddings
emb_scaled = emb_scaler.transform(emb_test)
Z = emb_pca.transform(emb_scaled).astype(np.float32)
print(f"PCA embedding shape: {Z.shape}")

# Fuse with priors
base_scores, prior_scores = fuse_scores(
    scores_test,
    meta_test["site"].to_numpy(),
    meta_test["hour_utc"].to_numpy(),
    prior_tables
)

# Compute probabilities for co-occurrence features
base_probs = sigmoid(base_scores)
print(f"Base scores shape: {base_scores.shape}")
print(f"Base probs shape: {base_probs.shape}")

# Determine which feature builder to use
pca_dim = Z.shape[1]
base_feature_dim = pca_dim + 7  # PCA + sequence features (raw, prior, base, prev, next, mean, max)
print(f"Base feature dim: {base_feature_dim}")

if n_features_expected == base_feature_dim:
    print("Using base features (no co-occurrence)")
    use_coocc_features = False
elif n_features_expected == base_feature_dim + 4:
    print("Using base + co-occurrence features (+4 dims)")
    use_coocc_features = True
else:
    print(f"Warning: Unexpected feature count {n_features_expected}, trying base + co-occurrence")
    use_coocc_features = True

# Apply LR probes
final_scores = base_scores.copy()

for cls_idx in tqdm(lr_probes.keys(), desc="LR Probes"):
    if use_coocc_features:
        X = build_features_with_coocc(
            Z, scores_test[:, cls_idx], prior_scores[:, cls_idx], base_scores[:, cls_idx],
            base_probs, cls_idx, cooccurrence_prob, pmi_matrix
        )
    else:
        X = build_features(
            Z, scores_test[:, cls_idx], prior_scores[:, cls_idx], base_scores[:, cls_idx]
        )
    
    pred = lr_probes[cls_idx].decision_function(X).astype(np.float32)
    final_scores[:, cls_idx] = (1.0 - PROBE_ALPHA) * base_scores[:, cls_idx] + PROBE_ALPHA * pred

print(f"Score range: {final_scores.min():.4f} to {final_scores.max():.4f}")

# Apply temperature scaling
predictions = sigmoid(final_scores, TEMPERATURE)

# =============================================================================
# CREATE SUBMISSION
# =============================================================================
print("\nCreating submission...")

submission = pd.DataFrame({'row_id': meta_test['row_id']})
for i, species in enumerate(PRIMARY_LABELS):
    submission[species] = predictions[:, i]

for species in PRIMARY_LABELS:
    submission[species] = submission[species].clip(1e-6, 1 - 1e-6)

submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False)

print(f"Submission saved: {OUTPUT_DIR / 'submission.csv'}")
print(f"Submission shape: {submission.shape}")
print(f"Prediction mean: {predictions.mean():.4f}")

print("\n" + "=" * 70)
print("Inference Complete!")
print("=" * 70)
print(f"Features: {n_features_expected}")
print(f"Temperature: {TEMPERATURE:.4f}")