LB Score is 0.867

TensorFlow version: 2.20.0
GPU available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')]
======================================================================
BirdCLEF+ 2026 - Training v6
======================================================================

Configuration:
Pseudo-labeling threshold: 0.95
Label smoothing: 0.1
Class-wise temperature: True

Loading competition data...
Classes: 234, Raw label rows: 739
Train audio files: 35549
After dedup: 739 labeled windows from 66 files

Loading Perch model (GPU)...

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1774167810.617435 24 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13757 MB memory: -> device: 0, name: Tesla T4, pci bus id: 0000:00:04.0, compute capability: 7.5
I0000 00:00:1774167810.620162 24 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 13757 MB memory: -> device: 1, name: Tesla T4, pci bus id: 0000:00:05.0, compute capability: 7.5

Mapped: 203, Unmapped: 31
Building frog proxies...
Texture: 42, Event: 33, Proxies: 3

---

## Running Perch on labeled training soundscapes...

I0000 00:00:1774167845.247550 80 device_compiler.h:196] Compiled cluster using XLA! This line is logged at most once for the lifetime of the process.

Total windows processed: 792
Y_TRAIN: (792, 234), positive rate: 1.68%

Building OOF...

OOF AUC (base): 0.502265

Preparing features...
PCA: 128, var: 0.8868
Classes with >= 4 positives: 60

---

## Training Initial LR Probes...

Trained 60 initial LR probes

---

## Pseudo-Labeling from train_audio...

Found 35549 train audio files
Selected 3000 files for processing

Processed 3000 audio segments

Pseudo-labeled samples: 2120
Total pseudo-labels added: 5616

---

## Retraining with Pseudo-labels...

Augmented data: 3792 samples (792 real + 3000 pseudo)

Trained 60 final LR probes

---

## Evaluating OOF...

OOF AUC (base): 0.502265
OOF AUC (final): 0.961922

---

## Temperature Scaling...

Computing class-wise temperatures...

Temperature range: 0.1000 - 1.9845
Mean temperature: 0.8496

Saving artifacts...

======================================================================
Training Complete!
======================================================================

Output Files:

- full_perch_arrays.npz
- full_perch_meta.parquet

Metrics:
OOF AUC (base): 0.502265
OOF AUC (final): 0.961922

======================================================================
BirdCLEF+ 2026 - Inference v6 (CPU Submission)
======================================================================

---

## Loading training artifacts...

Loaded arrays from: /kaggle/input/notebooks/blamerx/birdclef-2026-training/full_perch_arrays.npz

Config: PCA=128, LR_C=0.7, Alpha=0.5
Class-wise temperature: True
Class temperatures: shape=(234,), mean=0.8496

Deserializing models...
LR probes: 60

---

## Loading Perch v2 model (CPU)...

Perch model loaded from: /kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1

---

## Running Inference

No test files. Using 20 train soundscapes.

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1774168738.755627 66 device_compiler.h:196] Compiled cluster using XLA! This line is logged at most once for the lifetime of the process.

scores_raw: (240, 234), emb_test: (240, 1536)

Building predictions...
Base scores shape: (240, 234)

Score range: -45.5921 to 49.3426

Creating submission...
Submission saved: /kaggle/working/submission.csv
Submission shape: (240, 235)
Prediction mean: 0.4789

======================================================================
Inference Complete!
======================================================================
