"""
================================================================================
BirdCLEF+ 2026 - Inference Script (CPU Only for Submission)
================================================================================

Uses trained Perch v2 embeddings + Linear Probes.

================================================================================
"""

# =============================================================================
# INSTALL TENSORFLOW 2.20
# =============================================================================
!pip install -q --no-deps /kaggle/input/notebooks/kdmitrie/bc26-tensorflow-2-20-0/wheel/tensorboard-2.20.0-py3-none-any.whl
!pip install -q --no-deps /kaggle/input/notebooks/kdmitrie/bc26-tensorflow-2-20-0/wheel/tensorflow-2.20.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# =============================================================================
# IMPORTS
# =============================================================================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gc
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

from tqdm.auto import tqdm
import joblib

warnings.filterwarnings("ignore")
tf.experimental.numpy.experimental_enable_numpy_behavior()

# =============================================================================
# CONFIG
# =============================================================================
BASE = Path("/kaggle/input/competitions/birdclef-2026")
MODEL_DIR = Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1")
ARTIFACTS_DIR = Path("/kaggle/input/notebooks/blamerx/birdclef-2026-training")

SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12
BATCH_FILES = 16
DRYRUN_N_FILES = 20

print("=" * 60)
print("BirdCLEF+ 2026 - Inference")
print("=" * 60)

# =============================================================================
# LOAD ARTIFACTS
# =============================================================================
print("\nLoading artifacts...")

PRIMARY_LABELS = pd.read_csv(ARTIFACTS_DIR / "labels.csv")["label"].tolist()
N_CLASSES = len(PRIMARY_LABELS)
label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}

with open(ARTIFACTS_DIR / "config.json", "r") as f:
    config = json.load(f)

BEST_PROBE = config["best_probe"]
BEST_FUSION = config["best_fusion"]

# Load arrays
BC_INDICES = np.load(ARTIFACTS_DIR / "bc_indices.npy")
MAPPED_POS = np.load(ARTIFACTS_DIR / "mapped_pos.npy")
UNMAPPED_POS = np.load(ARTIFACTS_DIR / "unmapped_pos.npy")
MAPPED_BC_INDICES = np.load(ARTIFACTS_DIR / "mapped_bc_indices.npy")
selected_proxy_pos = np.load(ARTIFACTS_DIR / "selected_proxy_pos.npy")
idx_active_texture = np.load(ARTIFACTS_DIR / "idx_active_texture.npy")
idx_active_event = np.load(ARTIFACTS_DIR / "idx_active_event.npy")
idx_mapped_active_texture = np.load(ARTIFACTS_DIR / "idx_mapped_active_texture.npy")
idx_mapped_active_event = np.load(ARTIFACTS_DIR / "idx_mapped_active_event.npy")
idx_selected_proxy_active_texture = np.load(ARTIFACTS_DIR / "idx_selected_proxy_active_texture.npy")
idx_selected_prioronly_active_texture = np.load(ARTIFACTS_DIR / "idx_selected_prioronly_active_texture.npy")
idx_selected_prioronly_active_event = np.load(ARTIFACTS_DIR / "idx_selected_prioronly_active_event.npy")
idx_unmapped_inactive = np.load(ARTIFACTS_DIR / "idx_unmapped_inactive.npy")

with open(ARTIFACTS_DIR / "proxy_map.json", "r") as f:
    proxy_loaded = json.load(f)
selected_proxy_pos_to_bc = {int(k): np.array(v, dtype=np.int32) for k, v in proxy_loaded.items()}

# Load models
prior_tables = joblib.load(ARTIFACTS_DIR / "prior_tables.pkl")
emb_scaler = joblib.load(ARTIFACTS_DIR / "emb_scaler.pkl")
emb_pca = joblib.load(ARTIFACTS_DIR / "emb_pca.pkl")
probe_models = joblib.load(ARTIFACTS_DIR / "probe_models.pkl")

print(f"Classes: {N_CLASSES}, Probes: {len(probe_models)}")

# =============================================================================
# LOAD PERCH
# =============================================================================
print("\nLoading Perch model...")

birdclassifier = tf.saved_model.load(str(MODEL_DIR))
infer_fn = birdclassifier.signatures["serving_default"]

print("Perch loaded")

# =============================================================================
# UTILITIES
# =============================================================================
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
    prev_v = np.concatenate([x[:, :1], x[:, :-1]], axis=1).reshape(-1)
    next_v = np.concatenate([x[:, 1:], x[:, -1:]], axis=1).reshape(-1)
    mean_v = np.repeat(x.mean(axis=1), N_WINDOWS)
    max_v = np.repeat(x.max(axis=1), N_WINDOWS)
    return prev_v, next_v, mean_v, max_v


def build_class_features(emb, raw, prior, base):
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
# PERCH INFERENCE
# =============================================================================
FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")


def parse_filename(name):
    m = FNAME_RE.match(name)
    if not m:
        return {"site": None, "hour_utc": -1}
    _, site, _, hms = m.groups()
    return {"site": site, "hour_utc": int(hms[:2])}


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
        iterator = tqdm(iterator, total=(n_files + BATCH_FILES - 1) // BATCH_FILES, desc="Perch")

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
# RUN INFERENCE
# =============================================================================
print("\n" + "-" * 60)
print("Running Inference")
print("-" * 60)

test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))

if len(test_paths) == 0:
    print(f"No test files. Using {DRYRUN_N_FILES} train soundscapes.")
    test_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:DRYRUN_N_FILES]
else:
    print(f"Test files: {len(test_paths)}")

meta_test, scores_raw, emb_test = infer_perch(test_paths)

print(f"scores_raw: {scores_raw.shape}, emb_test: {emb_test.shape}")

# =============================================================================
# BUILD PREDICTIONS
# =============================================================================
print("\nBuilding predictions...")

# Fuse with priors
base_scores, prior_scores = fuse_scores(
    scores_raw, meta_test["site"].to_numpy(), meta_test["hour_utc"].to_numpy(), prior_tables
)

# Transform embeddings
emb_scaled = emb_scaler.transform(emb_test)
Z = emb_pca.transform(emb_scaled).astype(np.float32)

# Apply probes
final_scores = base_scores.copy()
for cls_idx, clf in tqdm(probe_models.items(), desc="Probes"):
    X = build_class_features(Z, scores_raw[:, cls_idx], prior_scores[:, cls_idx], base_scores[:, cls_idx])
    pred = clf.decision_function(X).astype(np.float32)
    final_scores[:, cls_idx] = (1.0 - BEST_PROBE["alpha"]) * base_scores[:, cls_idx] + BEST_PROBE["alpha"] * pred

print(f"Score range: {final_scores.min():.4f} to {final_scores.max():.4f}")

# =============================================================================
# SUBMISSION
# =============================================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


submission = pd.DataFrame(sigmoid(final_scores), columns=PRIMARY_LABELS)
submission.insert(0, "row_id", meta_test["row_id"].values)
submission[PRIMARY_LABELS] = submission[PRIMARY_LABELS].astype(np.float32)

# Validate
expected_rows = len(test_paths) * N_WINDOWS
assert len(submission) == expected_rows
assert submission.columns.tolist() == ["row_id"] + PRIMARY_LABELS
assert not submission.isna().any().any()

submission.to_csv("submission.csv", index=False)

print("\n" + "=" * 60)
print("Inference Complete!")
print("=" * 60)
print(f"Submission shape: {submission.shape}")
print(submission.iloc[:3, :8])
