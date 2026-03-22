# LB Score is 0.899

======================================================================
BirdCLEF+ 2026 - Training v3 (GPU + Advanced Features)
======================================================================
USE_TRAIN_AUDIO: True
USE_PSEUDO_LABELING: True

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
I0000 00:00:1774073826.925167 24 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13757 MB memory: -> device: 0, name: Tesla T4, pci bus id: 0000:00:04.0, compute capability: 7.5
I0000 00:00:1774073826.928224 24 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 13757 MB memory: -> device: 1, name: Tesla T4, pci bus id: 0000:00:05.0, compute capability: 7.5

Perch model loaded. Classes: 14795

Mapping competition labels to Perch labels...
Mapped: 203, Unmapped: 31

Building frog proxies for unmapped amphibians...
Texture classes: 42, Event classes: 33
Frog proxies: 3

---

## Running Perch on labeled training soundscapes...

I0000 00:00:1774073859.490255 80 device_compiler.h:196] Compiled cluster using XLA! This line is logged at most once for the lifetime of the process.

Total windows processed: 739 (filtered to labeled windows only)
scores_raw: (739, 234), emb_all: (739, 1536)
Y_TRAIN: (739, 234), positive rate: 1.8054%

---

## Running Perch on train_audio (short clips)...

Valid train_audio files: 35549

Processed 5000/35549 files

Processed 10000/35549 files

Processed 15000/35549 files

Processed 20000/35549 files

Processed 25000/35549 files

Processed 30000/35549 files

Processed 35000/35549 files

Processed 35549/35549 files

train_audio: 35549 clips
scores_audio: (35549, 234), emb_audio: (35549, 1536)
Species with train_audio data: 206

---

## Combining soundscape + train_audio data...

Combined: 36288 samples

- Soundscapes: 739
- Train audio: 35549
  Species with data: 234

Building OOF (soundscapes only)...

OOF AUC (soundscapes): 0.500290

---

## Training initial probes...

PCA: 64, var: 0.3896

Trained 212 LR probes, 212 MLP probes

======================================================================
PSEUDO-LABELING UNLABELED SOUNDSCAPES
======================================================================

Processing 10592 unlabeled soundscape files...

Unlabeled embeddings: (127104, 1536)

Generating pseudo-labels...

Pseudo-labels generated: 14241 windows
Total pseudo-label count: 15458

Combined with pseudo-labels: 50529 samples

---

## Retraining probes with pseudo-labels...

Retrained 212 LR probes, 212 MLP probes

---

## Evaluating OOF with probes...

--- Soundscape OOF Results ---
OOF AUC (base): 0.500290
OOF AUC (LR): 0.918820
OOF AUC (MLP): 0.851622
OOF AUC (final): 0.916584

---

## Saving artifacts...

======================================================================
Training Complete!
======================================================================

Data Summary:
Soundscapes (labeled): 739 windows
Train Audio: 35549 clips
Pseudo-labeled: 14241 windows

Output Files:

- full_perch_arrays.npz → embeddings + scores + models + indices
- full_perch_meta.parquet → soundscape metadata

Metrics:
Soundscape OOF AUC (final): 0.916584

======================================================================
BirdCLEF+ 2026 - Inference v3 (CPU Submission)
======================================================================

---

## Loading training artifacts...

Loaded arrays from: /kaggle/input/notebooks/blamerx/birdclef-2026-training/full_perch_arrays.npz
Config: {'best_probe': {'pca_dim': 64, 'min_pos': 8, 'C': 0.5, 'alpha': 0.4, 'mlp_hidden': [128], 'mlp_activation': 'relu', 'mlp_max_iter': 300, 'mlp_early_stopping': True, 'mlp_validation_fraction': 0.15, 'mlp_n_iter_no_change': 15, 'mlp_learning_rate_init': 0.001, 'mlp_alpha': 0.01, 'mlp_random_state': 42}, 'best_ensemble': {'method': 'min', 'lr_weight': 0.5, 'mlp_weight': 0.5}, 'best_fusion': {'lambda_event': 0.4, 'lambda_texture': 1.0, 'lambda_proxy_texture': 0.8, 'smooth_texture': 0.35}, 'pseudo_params': {'threshold_high': 0.95, 'threshold_low': 0.05, 'max_per_class': 500, 'min_per_class': 10}, 'n_classes': 234, 'n_windows': 12, 'n_soundscapes': 739, 'n_train_audio': 35549, 'use_pseudo_labeling': True, 'use_train_audio': True}
Classes: 234
Labels loaded: 234
Mapped species: 203
Unmapped species: 31
Frog proxies: 3

Deserializing models...
LR probes: 212
MLP probes: 212
PCA components: 64

---

## Loading Perch v2 model (CPU)...

Perch model loaded from: /kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1

---

## Running Inference

No test files. Using 20 train soundscapes.

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1774082737.281540 66 device_compiler.h:196] Compiled cluster using XLA! This line is logged at most once for the lifetime of the process.

scores_raw: (240, 234), emb_test: (240, 1536)

Building predictions...

Score range: -30.1725 to 8.8029
Prediction range: 0.0000 to 0.9998

Creating submission...
Submission saved: /kaggle/working/submission.csv

Submission shape: (240, 235)
Prediction mean: 0.0503
Prediction std: 0.1325

======================================================================
Inference Complete!
======================================================================
