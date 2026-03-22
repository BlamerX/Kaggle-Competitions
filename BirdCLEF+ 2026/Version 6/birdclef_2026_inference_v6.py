"""
================================================================================
BirdCLEF+ 2026 - Inference Script v6 (CPU Submission)
================================================================================

Uses pre-trained LR probes from v6 GPU training phase.
Runs on CPU within 90-minute Kaggle submission constraint.

================================================================================
"""

# =============================================================================
# INSTALL TENSORFLOW 2.20 (Required for Perch v2 CPU)
# =============================================================================
!pip install -q --no-deps /kaggle/input/notebooks/kdmitrie/bc26-tensorflow-2-20-0/wheel/tensorboard-2.20.0-py3-none-any.whl
!pip install -q --no-deps /kaggle/input/notebooks/kdmitrie/bc26-tensorflow-2-20-0/wheel/tensorflow-2.20.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12
BATCH_FILES = 50

ARRAYS_FILE = MODELS_DIR / "full_perch_arrays.npz"

print("=" * 70)
print("BirdCLEF+ 2026 - Inference v6 (CPU Submission)")
print("=" * 70)

# =============================================================================
# LOAD TRAINING ARTIFACTS
# =============================================================================
print("\n" + "-" * 70)
print("Loading training artifacts...")
print("-" * 70)

arrays = np.load(ARRAYS_FILE, allow_pickle=True)
print(f"Loaded arrays from: {ARRAYS_FILE}")

config = json.loads(str(arrays['config']))

N_CLASSES = config['n_classes']
N_WINDOWS = config['n_windows']
BEST_FUSION = config['best_fusion']
PROBE_ALPHA = config.get('probe_alpha', 0.5)
USE_CLASS_TEMPERATURE = config.get('use_class_temperature', False)

print(f"\nConfig: PCA={config['pca_dim']}, LR_C={config['lr_c']}, Alpha={PROBE_ALPHA}")
print(f"Class-wise temperature: {USE_CLASS_TEMPERATURE}")

PRIMARY_LABELS = arrays['labels'].tolist()

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

proxy_map_keys = arrays['proxy_map_keys']
proxy_map_vals = arrays['proxy_map_vals']
selected_proxy_pos_to_bc = {int(k): np.array(v, dtype=np.int32) for k, v in zip(proxy_map_keys, proxy_map_vals)}

class_temperatures = arrays.get('class_temperatures', None)
if class_temperatures is None:
    class_temperatures = np.ones(N_CLASSES, dtype=np.float32)
print(f"Class temperatures: shape={class_temperatures.shape}, mean={class_temperatures.mean():.4f}")

def deserialize_pickle(arr):
    buf = BytesIO(arr.tobytes())
    return pickle.load(buf)

print("\nDeserializing models...")
lr_probes = deserialize_pickle(arrays['lr_probes_bytes'])
emb_scaler = deserialize_pickle(arrays['emb_scaler_bytes'])
emb_pca = deserialize_pickle(arrays['emb_pca_bytes'])
prior_tables = deserialize_pickle(arrays['prior_tables_bytes'])

print(f"LR probes: {len(lr_probes)}")

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
def sigmoid_with_temperature(x, temperature):
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
    prev_b, next_b, mean_b, max_b = seq_features_1d(base)
    return np.concatenate([
        emb, raw[:, None], prior[:, None], base[:, None],
        prev_b[:, None], next_b[:, None], mean_b[:, None], max_b[:, None]
    ], axis=1).astype(np.float32)

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

meta_test, scores_test, emb_test = infer_perch_soundscapes(test_paths)
print(f"scores_raw: {scores_test.shape}, emb_test: {emb_test.shape}")

# =============================================================================
# BUILD PREDICTIONS
# =============================================================================
print("\nBuilding predictions...")

emb_scaled = emb_scaler.transform(emb_test)
Z = emb_pca.transform(emb_scaled).astype(np.float32)

base_scores, prior_scores = fuse_scores(
    scores_test,
    meta_test["site"].to_numpy(),
    meta_test["hour_utc"].to_numpy(),
    prior_tables
)

print(f"Base scores shape: {base_scores.shape}")

final_scores = base_scores.copy()

for cls_idx in tqdm(lr_probes.keys(), desc="LR Probes"):
    X = build_features(
        Z, scores_test[:, cls_idx], prior_scores[:, cls_idx], base_scores[:, cls_idx]
    )
    pred = lr_probes[cls_idx].decision_function(X).astype(np.float32)
    final_scores[:, cls_idx] = (1.0 - PROBE_ALPHA) * base_scores[:, cls_idx] + PROBE_ALPHA * pred

print(f"Score range: {final_scores.min():.4f} to {final_scores.max():.4f}")

# Apply temperature scaling
if USE_CLASS_TEMPERATURE:
    predictions = np.zeros_like(final_scores)
    for cls_idx in range(N_CLASSES):
        predictions[:, cls_idx] = sigmoid_with_temperature(
            final_scores[:, cls_idx], 
            class_temperatures[cls_idx]
        )
else:
    predictions = sigmoid_with_temperature(final_scores, class_temperatures.mean())

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