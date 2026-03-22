# BirdCLEF+ 2026 - File Structure Reference
## ⚠️ CRITICAL INFORMATION - SAVE THIS!

---

## 📁 Base Path
```
/kaggle/input/competitions/birdclef-2026
```

---

## 🔑 KEY DISCOVERY: Filename Contains Subdirectory!

### The train.csv `filename` column ALREADY includes the path!

```
train.csv sample:
                  filename    collection    primary_label
0  1161364/iNat1216197.ogg          iNat         1161364
1  1161364/iNat1114648.ogg          iNat         1161364
```

### ✅ CORRECT Path Pattern:
```python
# CORRECT - filename already has subdirectory!
audio_path = f"{BASE_PATH}/train_audio/{row['filename']}"
# Example: /kaggle/input/competitions/birdclef-2026/train_audio/1161364/iNat1216197.ogg

# WRONG - DO NOT add collection subdirectory!
audio_path = f"{BASE_PATH}/train_audio/{row['collection']}/{row['filename']}"  # ❌ WRONG
```

---

## 📊 Dataset Statistics

| Item | Count |
|------|-------|
| train.csv rows | 35,549 |
| Species (subdirectories) | 206 |
| train_audio files | 35,549 .ogg |
| train_soundscapes files | 10,658 .ogg |
| Total audio files | 46,207 |
| Collections | iNat, XC |
| **Competition Classes** | **234** |

---

## 📂 Directory Structure

```
/kaggle/input/competitions/birdclef-2026/
├── recording_location.txt
├── sample_submission.csv
├── taxonomy.csv
├── train.csv
├── train_soundscapes_labels.csv
├── train_audio/
│   ├── 1161364/
│   │   ├── iNat1216197.ogg
│   │   ├── iNat1114648.ogg
│   │   └── ... (11 files)
│   ├── 116570/
│   │   └── iNat1460166.ogg
│   ├── 1176823/
│   │   └── ... (12 files)
│   └── ... (206 subdirectories total)
├── train_soundscapes/
│   ├── BC2026_Train_0001_S08_20250606_030007.ogg
│   ├── BC2026_Train_0002_S08_20250607_030007.ogg
│   └── ... (10,658 files directly - NO subdirectories!)
└── test_soundscapes/
    └── (populated during submission - ~600 files)
```

---

## 🎵 Audio File Details

| Property | Value |
|----------|-------|
| Format | .ogg |
| Sample Rate | 32 kHz |
| train_audio duration | Variable (short recordings) |
| train_soundscapes duration | 1 minute each |
| test_soundscapes duration | 1 minute each |

---

## 📋 Train CSV Columns
```
['primary_label', 'secondary_labels', 'type', 'latitude', 'longitude',
 'scientific_name', 'common_name', 'class_name', 'inat_taxon_id',
 'author', 'license', 'rating', 'url', 'filename', 'collection']
```

---

## 🔧 Model Paths

### Perch v2
| Environment | Path |
|-------------|------|
| GPU Training | `/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2/2` |
| CPU Inference | `/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1` |

### Training Output (v4)
```
/kaggle/working/  (or Kaggle Dataset)
├── full_perch_arrays.npz    → All artifacts in single file
│   ├── embeddings           → (N, 1536) float32
│   ├── scores_raw           → (N, 234) float32
│   ├── oof_base             → (N, 234) float32
│   ├── oof_prior            → (N, 234) float32
│   ├── oof_lr               → (N, 234) float32
│   ├── emb_audio            → (M, 1536) float32
│   ├── scores_audio         → (M, 234) float32
│   ├── bc_indices           → (234,) int32
│   ├── mapped_pos           → mapped class indices
│   ├── unmapped_pos         → unmapped class indices
│   ├── mapped_bc_indices    → Perch indices for mapped classes
│   ├── selected_proxy_pos   → frog proxy indices
│   ├── idx_active_texture   → texture class indices
│   ├── idx_active_event     → event class indices
│   ├── idx_mapped_active_texture
│   ├── idx_mapped_active_event
│   ├── idx_selected_proxy_active_texture
│   ├── idx_selected_prioronly_active_texture
│   ├── idx_selected_prioronly_active_event
│   ├── idx_unmapped_inactive
│   ├── proxy_map_keys       → proxy label keys
│   ├── proxy_map_vals       → proxy label values
│   ├── config               → JSON config
│   ├── labels               → species labels
│   ├── lr_probes_bytes      → pickled LR models
│   ├── mlp_probes_bytes     → pickled MLP models (empty in v4)
│   ├── emb_scaler_bytes     → pickled StandardScaler
│   ├── emb_pca_bytes        → pickled PCA
│   └── prior_tables_bytes   → pickled prior tables
└── full_perch_meta.parquet  → metadata
    └── row_id, site, hour_utc
```

---

## 🔧 Correct Code Implementation

### Loading Audio Files:
```python
import pandas as pd
import soundfile as sf

BASE_PATH = "/kaggle/input/competitions/birdclef-2026"
train_df = pd.read_csv(f"{BASE_PATH}/train.csv")

# Get audio path - CORRECT WAY
for idx, row in train_df.iterrows():
    audio_path = f"{BASE_PATH}/train_audio/{row['filename']}"
    audio, sr = sf.read(audio_path)
    # Process audio...
```

### Loading Perch Model:
```python
import tensorflow as tf

# GPU (Training)
MODEL_DIR = "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2/2"
birdclassifier = tf.saved_model.load(MODEL_DIR)
infer_fn = birdclassifier.signatures["serving_default"]

# CPU (Inference)
PERCH_CPU = "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1"
perch_model = tf.saved_model.load(PERCH_CPU)
infer_fn = perch_model.signatures["serving_default"]

# Usage
outputs = infer_fn(inputs=tf.convert_to_tensor(audio_batch))
logits = outputs["label"].numpy()      # (batch, num_perch_classes)
embeddings = outputs["embedding"].numpy()  # (batch, 1536)
```

### Loading Training Artifacts:
```python
import numpy as np
import json
import pickle
from io import BytesIO

MODELS_DIR = "/kaggle/input/notebooks/blamerx/birdclef-2026-training"
arrays = np.load(f"{MODELS_DIR}/full_perch_arrays.npz", allow_pickle=True)

# Load config
config = json.loads(str(arrays['config']))
N_CLASSES = config['n_classes']
BEST_PROBE = config['best_probe']
BEST_FUSION = config['best_fusion']

# Load labels
PRIMARY_LABELS = arrays['labels'].tolist()

# Deserialize models
def deserialize_pickle(arr):
    buf = BytesIO(arr.tobytes())
    return pickle.load(buf)

lr_probes = deserialize_pickle(arrays['lr_probes_bytes'])
scaler = deserialize_pickle(arrays['emb_scaler_bytes'])
pca = deserialize_pickle(arrays['emb_pca_bytes'])
priors = deserialize_pickle(arrays['prior_tables_bytes'])
```

---

## 📝 Submission Format

```python
# submission.csv format
row_id,species1,species2,...,species234
BC2026_Test_0001_S05_20250227_010002_5,0.001,0.002,...,0.001
BC2026_Test_0001_S05_20250227_010002_10,0.001,0.003,...,0.002
...

# Each row_id = {filename}_{end_time}
# end_time: 5, 10, 15, ..., 60 (12 rows per file)
```

---

## 📊 Version-Specific Parameters

### v4 (Current)
```python
BEST_PROBE = {
    "pca_dim": 128,          # Increased from 64
    "min_pos": 8,
    "C": 0.50,
    "alpha": 0.40,
}

BEST_FUSION = {
    "lambda_event": 0.4,
    "lambda_texture": 1.0,
    "lambda_proxy_texture": 0.8,
    "smooth_texture": 0.35,
}

SAMPLE_WEIGHTS = {
    "soundscape_weight": 10.0,
    "train_audio_weight": 1.0,
}

BATCH_FILES = 4   # GPU training
BATCH_FILES = 50  # CPU inference (100 causes OOM)
```

---

## 🏷️ Species Classification

### Class Types (from taxonomy.csv)
| class_name | Type | Examples |
|------------|------|----------|
| Aves | Event (birds) | Most species |
| Amphibia | Texture (frogs) | Frogs, toads |
| Insecta | Texture (insects) | Cicadas, crickets |
| Mammalia | Event | Bats, monkeys |
| Reptilia | Event | Lizards, snakes |

### Label Mapping
- **Mapped**: Species with Perch label (direct prediction)
- **Unmapped**: Species without Perch label (use proxy or prior only)
- **Proxy**: Unmapped amphibians with genus match in Perch

---

**Last Updated**: 2026-03-21
