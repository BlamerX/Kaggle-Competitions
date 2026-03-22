# LB Score is 0.828

======================================================================
BirdCLEF+ 2026 - Training v4 (Simplified)
======================================================================
USE_TRAIN_AUDIO: True
USE_PSEUDO_LABELING: False
PCA dimensions: 128
Model: LR ONLY (MLP dropped - was hurting performance)

---

## Loading competition data...

Classes: 234
train.csv rows: 35549
Labeled soundscape windows: 739
Labeled soundscape files: 66
Unlabeled soundscape files: 10592
Label density in soundscapes: 1.8054%

---

## Loading Perch v2 model (GPU)...

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1774096450.288295 24 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13757 MB memory: -> device: 0, name: Tesla T4, pci bus id: 0000:00:04.0, compute capability: 7.5
I0000 00:00:1774096450.291087 24 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 13757 MB memory: -> device: 1, name: Tesla T4, pci bus id: 0000:00:05.0, compute capability: 7.5

Perch model loaded. Classes: 14795

Mapping competition labels to Perch labels...
Mapped: 203, Unmapped: 31

Building frog proxies for unmapped amphibians...
Texture classes: 42, Event classes: 33
Frog proxies: 3

---

## Running Perch on labeled training soundscapes...

Perch Soundscapes: 100%
 17/17 [01:09<00:00,  7.11s/it]

I0000 00:00:1774096481.501935 80 device_compiler.h:196] Compiled cluster using XLA! This line is logged at most once for the lifetime of the process.

Total windows processed: 739
scores_raw: (739, 234), emb_all: (739, 1536)
Y_TRAIN: (739, 234), positive rate: 1.8054%

---

## Running Perch on train_audio (short clips)...

Valid train_audio files: 35549

Perch Short: 100%
 105/105 [03:33<00:00,  5.99s/it]

Processed 5000/35549 files

Perch Short: 100%
 105/105 [03:36<00:00,  1.56s/it]

Processed 10000/35549 files

Perch Short: 100%
 105/105 [03:20<00:00,  1.72s/it]

Processed 15000/35549 files

Perch Short: 100%
 105/105 [03:14<00:00,  1.61s/it]

Processed 20000/35549 files

Perch Short: 100%
 105/105 [03:23<00:00,  1.65s/it]

Processed 25000/35549 files

Perch Short: 100%
 105/105 [03:13<00:00,  1.48s/it]

Processed 30000/35549 files

Perch Short: 100%
 105/105 [03:25<00:00,  1.51s/it]

Processed 35000/35549 files

Perch Short: 100%
 12/12 [00:40<00:00,  7.71s/it]

Processed 35549/35549 files

train_audio: 35549 clips

Building OOF (soundscapes only)...

OOF: 100%
 5/5 [00:00<00:00, 90.90it/s]

OOF AUC (soundscapes): 0.500290

---

## Training LR probes (v4: Soundscape-only PCA)...

PCA trained on SOUNSCAPE embeddings only
PCA: 128 components, variance: 0.8936
Combined training data: 36288 samples

Applying sample weights...
Soundscape samples: 739 (weight=10.0x)
Train audio samples: 35549 (weight=1.0x)

LR Probes: 100%
 212/212 [10:07<00:00,  2.87s/it]

Trained 212 LR probes (MLP skipped - was hurting performance)

---

## Evaluating OOF with LR probes...

--- Soundscape OOF Results ---
OOF AUC (base): 0.500290
OOF AUC (LR): 0.933102

---

## Saving artifacts...

======================================================================
Training Complete!
======================================================================

Data Summary:
Soundscapes (labeled): 739 windows
Train Audio: 35549 clips

Output Files:

- full_perch_arrays.npz
- full_perch_meta.parquet

Metrics:
OOF AUC (LR): 0.933102

======================================================================
BirdCLEF+ 2026 - Inference v4 (CPU Submission)
======================================================================

---

## Loading training artifacts...

Loaded arrays from: /kaggle/input/notebooks/blamerx/birdclef-2026-training/full_perch_arrays.npz
Config: {'best_probe': {'pca_dim': 128, 'min_pos': 8, 'C': 0.5, 'alpha': 0.4}, 'best_ensemble': {'method': 'none', 'lr_weight': 1.0, 'mlp_weight': 0.0}, 'best_fusion': {'lambda_event': 0.4, 'lambda_texture': 1.0, 'lambda_proxy_texture': 0.8, 'smooth_texture': 0.35}, 'sample_weights': {'soundscape_weight': 10.0, 'train_audio_weight': 1.0}, 'n_classes': 234, 'n_windows': 12, 'n_soundscapes': 739, 'n_train_audio': 35549, 'use_pseudo_labeling': False, 'use_train_audio': True}
Classes: 234
Labels loaded: 234
Mapped species: 203
Unmapped species: 31
Frog proxies: 3

Deserializing models...
LR probes: 212
MLP probes: 0
PCA components: 128

---

## Loading Perch v2 model (CPU)...

Perch model loaded from: /kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1

---

## Running Inference

No test files. Using 20 train soundscapes.

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1774106123.122564 66 device_compiler.h:196] Compiled cluster using XLA! This line is logged at most once for the lifetime of the process.

scores_raw: (240, 234), emb_test: (240, 1536)

Building predictions...

Score range: -74.6314 to 34.2277
Prediction range: 0.0000 to 1.0000

Creating submission...
Submission saved: /kaggle/working/submission.csv

Submission shape: (240, 235)
Prediction mean: 0.0415
Prediction std: 0.1593

======================================================================
Inference Complete!
======================================================================
