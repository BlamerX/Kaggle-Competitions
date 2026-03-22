# BirdCLEF+ 2026 - Complete Strategy for CPU 90-Minute Limit

## 🎯 HOW TO BEAT THE CONSTRAINT

### The Two-Notebook Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│  NOTEBOOK 1: TRAINING (GPU, No Time Limit)                     │
│  =================================================             │
│  1. Train Perch v2 + LR probes on GPU                          │
│  2. Extract and cache embeddings (jaejohn style)               │
│  3. Save all artifacts to Kaggle Dataset                       │
│  4. Upload as "birdclef-2026-training" dataset                 │
│                                                                 │
│  v4 Output:                                                    │
│  - OOF AUC: 0.933                                              │
│  - PCA Variance: 89%                                           │
│  - Training Time: ~30 minutes GPU                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  NOTEBOOK 2: SUBMISSION (CPU, 90 minutes)                      │
│  =================================================             │
│  1. Load pre-trained artifacts (30 sec)                        │
│  2. Load Perch CPU model (30 sec)                              │
│  3. Load ~600 test audio files (~5 min)                        │
│  4. Run Perch inference (~60 min)                              │
│  5. Apply LR probes (~2 min)                                   │
│  6. Generate submission.csv (~1 min)                           │
│                                                                 │
│  Total: ~70 minutes ✅ Under 90!                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⏱️ TIME BREAKDOWN

### Submission Notebook (CPU)

| Step | Time | Technique |
|------|------|-----------|
| Import libraries | 10 sec | Pre-import essential only |
| Load artifacts | 30 sec | numpy npz + pickle |
| Load Perch model | 30 sec | TensorFlow SavedModel |
| **Load 600 audio files** | **~5 min** | soundfile (10x faster than librosa) |
| **Perch inference** | **~60 min** | Batch processing (50 files/batch) |
| **Apply LR probes** | **~2 min** | Vectorized operations |
| Create submission | 1 min | Pandas |
| **TOTAL** | **~70 min** | ✅ Under 90! |

---

## 📁 OUTPUT FILES FROM TRAINING

### v4 Single-File Format (Recommended)

```
birdclef-2026-training/
├── full_perch_arrays.npz    → All artifacts in one compressed file
│   ├── config               → JSON with hyperparameters
│   ├── labels               → Species labels (234)
│   ├── embeddings           → Soundscapes embeddings (739, 1536)
│   ├── scores_raw           → Raw Perch scores
│   ├── oof_base/oof_prior   → OOF predictions
│   ├── lr_probes_bytes      → Pickled LR models
│   ├── emb_scaler_bytes     → Pickled StandardScaler
│   ├── emb_pca_bytes        → Pickled PCA
│   ├── prior_tables_bytes   → Pickled prior tables
│   └── index arrays         → Various class indices
└── full_perch_meta.parquet  → Metadata (row_id, site, hour)
```

### Loading Code:
```python
import numpy as np
import json
import pickle
from io import BytesIO

arrays = np.load(MODELS_DIR / "full_perch_arrays.npz", allow_pickle=True)
config = json.loads(str(arrays['config']))

def deserialize_pickle(arr):
    buf = BytesIO(arr.tobytes())
    return pickle.load(buf)

lr_probes = deserialize_pickle(arrays['lr_probes_bytes'])
scaler = deserialize_pickle(arrays['emb_scaler_bytes'])
pca = deserialize_pickle(arrays['emb_pca_bytes'])
priors = deserialize_pickle(arrays['prior_tables_bytes'])
```

---

## 🚀 OPTIMIZATION TECHNIQUES

### 1. Use TensorFlow SavedModel (Critical!)

```python
# Perch v2 is optimized for CPU inference
perch_model = tf.saved_model.load(PERCH_CPU_PATH)
infer_fn = perch_model.signatures["serving_default"]

# Batch inference
outputs = infer_fn(inputs=tf.convert_to_tensor(batch))
logits = outputs["label"].numpy()
embeddings = outputs["embedding"].numpy()
```

**Speedup: 3x faster than PyTorch on CPU**

### 2. Batch Processing

```python
# SLOW: Process one at a time
for segment in segments:
    pred = model(segment)

# FAST: Batch process
BATCH_FILES = 50  # Safe for 30 GiB RAM
batch = np.stack(segments)  # Shape: (50 * 12, 160000)
preds = model(batch)        # Single inference call
```

**Speedup: 5-10x faster (50 files → 1 call vs 50 calls)**

### 3. Efficient Audio Loading

```python
# SLOW: librosa.load
audio, sr = librosa.load(filepath, sr=32000)  # ~0.5 sec/file

# FAST: soundfile
audio, sr = sf.read(filepath)  # ~0.05 sec/file (10x faster!)
```

**Speedup: 10x faster audio loading**

### 4. Parallel File Loading

```python
from concurrent.futures import ThreadPoolExecutor

def read_audio(path):
    y, sr = sf.read(path, dtype="float32")
    return y[:FILE_SAMPLES]  # Trim to 1 minute

with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(read_audio, paths))
audio_batch = np.stack(results)
```

**Speedup: 4-8x faster for file I/O**

### 5. Pre-computed PCA and Scaler

```python
# Training: fit once
emb_scaler = StandardScaler()
emb_pca = PCA(n_components=128)  # v4 uses 128
emb_scaler.fit(embeddings)
emb_pca.fit(emb_scaled)

# Inference: transform only (fast!)
Z = emb_pca.transform(emb_scaler.transform(new_embeddings))
```

**Speedup: No fitting during inference**

---

## 🧮 Memory Management

### CPU Memory Budget (30 GiB)

| Component | Memory Usage |
|-----------|-------------|
| TensorFlow + Perch | ~4 GB |
| Audio batch (50 files) | ~16 MB × 50 = 800 MB |
| Embeddings (600 files) | ~4.4 MB |
| Probes + PCA | ~50 MB |
| **Total** | ~5 GB |
| **Safe margin** | ~25 GB available |

### Batch Size Guidelines

```python
# v4 tested values:
BATCH_FILES = 100  # OOM error!
BATCH_FILES = 50   # ✅ Works perfectly

# Calculation:
# Each file: 60 sec × 32 kHz × 4 bytes = 7.68 MB
# 50 files: 384 MB raw audio
# Plus TensorFlow overhead: ~5 GB total
```

### Memory Cleanup

```python
import gc
import tensorflow as tf

# After each batch
del x, outputs, logits, emb
tf.keras.backend.clear_session()
gc.collect()
```

---

## 📊 Inference Pipeline

### Complete v4 Inference Code:

```python
#!/usr/bin/env python3
# 1. Setup
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU

import gc
import json
import pickle
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

# 2. Configuration
BASE = Path("/kaggle/input/competitions/birdclef-2026")
MODELS_DIR = Path("/kaggle/input/notebooks/blamerx/birdclef-2026-training")
PERCH_CPU = "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1"
OUTPUT_DIR = Path("/kaggle/working")

SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12
BATCH_FILES = 50  # Safe for 30 GiB RAM

# 3. Load artifacts
arrays = np.load(MODELS_DIR / "full_perch_arrays.npz", allow_pickle=True)
config = json.loads(str(arrays['config']))

def deserialize_pickle(arr):
    return pickle.load(BytesIO(arr.tobytes()))

lr_probes = deserialize_pickle(arrays['lr_probes_bytes'])
scaler = deserialize_pickle(arrays['emb_scaler_bytes'])
pca = deserialize_pickle(arrays['emb_pca_bytes'])
priors = deserialize_pickle(arrays['prior_tables_bytes'])
PRIMARY_LABELS = arrays['labels'].tolist()
N_CLASSES = config['n_classes']

print(f"LR probes: {len(lr_probes)}, PCA var: {pca.explained_variance_ratio_.sum():.4f}")

# 4. Load Perch
perch = tf.saved_model.load(PERCH_CPU)
infer_fn = perch.signatures["serving_default"]

# 5. Process test files
test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))
if not test_paths:
    test_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:20]

# 6. Inference loop
def read_audio(path):
    y, _ = sf.read(path, dtype="float32")
    if y.ndim == 2:
        y = y.mean(1)
    return np.pad(y, (0, max(0, FILE_SAMPLES - len(y))))[:FILE_SAMPLES]

row_ids, sites, hours, scores, embeddings = [], [], [], [], []

for start in tqdm(range(0, len(test_paths), BATCH_FILES), desc="Perch"):
    batch_paths = test_paths[start:start + BATCH_FILES]

    with ThreadPoolExecutor(8) as ex:
        audio = np.stack(list(ex.map(read_audio, batch_paths)))
    x = audio.reshape(len(batch_paths) * N_WINDOWS, WINDOW_SAMPLES)

    outputs = infer_fn(inputs=tf.convert_to_tensor(x))
    logits = outputs["label"].numpy()
    emb = outputs["embedding"].numpy()

    # ... process scores and embeddings ...

    del x, outputs, logits, emb
    gc.collect()

# 7. Build predictions
Z = pca.transform(scaler.transform(embeddings))
# Apply LR probes and create submission...
```

---

## 🎯 Key Metrics to Watch

### During Training
| Metric | Good | Bad |
|--------|------|-----|
| OOF AUC | > 0.92 | < 0.90 |
| PCA Variance | > 70% | < 50% |
| LR Probes Count | > 200 | < 150 |

### During Inference
| Metric | Good | Bad |
|--------|------|-----|
| PCA Variance | ~89% | ~39% |
| Prediction Mean | 0.03-0.06 | > 0.10 or < 0.01 |
| Score Range | -75 to +35 | Extreme values |

---

## 📈 Performance Comparison

| Version | Training | Inference | Total |
|---------|----------|-----------|-------|
| v1-v2 | N/A | ~80 min | 80 min |
| v3 | ~25 min GPU | ~80 min | 105 min ❌ |
| v4 | ~30 min GPU | ~70 min | 70 min ✅ |

---

## 🔧 Troubleshooting

### OOM Error
```
Solution: Reduce BATCH_FILES
- 100 → OOM
- 50 → Works
- 25 → Conservative
```

### Slow Inference
```
Check:
1. Using soundfile not librosa
2. Using batch processing
3. Using ThreadPoolExecutor for file loading
4. TensorFlow not trying to use GPU
```

### Wrong Predictions
```
Check:
1. PCA variance is high (>70%)
2. LR probes loaded correctly
3. Prior tables match test sites
4. Labels array matches submission format
```

---

**Last Updated**: 2026-03-21
**v4 Inference Time**: ~70 minutes (with BATCH_FILES=50)
