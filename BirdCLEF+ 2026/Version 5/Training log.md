# LB Score is 0.904

TensorFlow version: 2.20.0
GPU available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')]
======================================================================
BirdCLEF+ 2026 - Training v5
======================================================================

1. Pseudo-labeling - threshold=0.85
2. Species Co-occurrence Features
3. Taxonomy-aware Grouping

Loading competition data...
Classes: 234, Raw label rows: 739

---

## IMPROVEMENT 3: Building Taxonomy Groupings...

Class groups: 5 (['Insecta', 'Reptilia', 'Amphibia', 'Mammalia', 'Aves']...)
Order groups: 1
Family groups: 1
After dedup: 739 labeled windows from 66 files

---

## IMPROVEMENT 2: Building Species Co-occurrence Matrix...

Co-occurrence matrix: (234, 234)
Top co-occurring pairs: [18 10 60 57 63]

Loading Perch model (GPU)...

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1774152633.850231 24 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13757 MB memory: -> device: 0, name: Tesla T4, pci bus id: 0000:00:04.0, compute capability: 7.5
I0000 00:00:1774152633.853266 24 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 13757 MB memory: -> device: 1, name: Tesla T4, pci bus id: 0000:00:05.0, compute capability: 7.5

Mapped: 203, Unmapped: 31
Building frog proxies...
Texture: 42, Event: 33, Proxies: 3

---

## Running Perch on labeled training soundscapes...

Perch GPU: 100%
 17/17 [01:05<00:00,  6.92s/it]

I0000 00:00:1774152667.615872 80 device_compiler.h:196] Compiled cluster using XLA! This line is logged at most once for the lifetime of the process.

Total windows processed: 792
Y_TRAIN: (792, 234), positive rate: 1.68%

Building OOF...

OOF: 100%
 5/5 [00:00<00:00, 71.31it/s]

OOF AUC (base): 0.502265

Preparing features...
PCA: 128, var: 0.8868

---

## IMPROVEMENT 2: Pre-computing Co-occurrence Features...

Co-occurrence features: (792, 234), PMI features: (792,)

---

## IMPROVEMENT 3: Pre-computing Taxonomy Features...

Taxonomy features: (792, 234), (792, 234)

---

## Training Initial LR Probes...

Classes with >= 4 positives: 60

Initial LR: 100%
 60/60 [00:01<00:00, 36.79it/s]

Trained 60 initial LR probes

---

## IMPROVEMENT 1: Pseudo-Labeling...

OOF Predictions: 100%
 60/60 [00:00<00:00, 1127.08it/s]

Pseudo-labeled samples: 4
Total pseudo-labels added: 4
Combined label density: 1.69% (was 1.68%)

---

## Retraining with Pseudo-labels...

Final LR: 100%
 60/60 [00:01<00:00, 37.32it/s]

Trained 60 final LR probes

Evaluating OOF...

Final Predictions: 100%
 60/60 [00:00<00:00, 1250.82it/s]

OOF AUC (base): 0.502265
OOF AUC (with pseudo): 0.961737

---

## Temperature Scaling...

Optimal temperature: 5.8471

Saving artifacts...

======================================================================
Training Complete!
======================================================================

Output Files:

- full_perch_arrays.npz
- full_perch_meta.parquet

1. Pseudo-labeling: 4 samples, 4 labels added
2. Co-occurrence Features: 234 features
3. Taxonomy Features: 468 features
4. Temperature: 5.8471

Metrics:
OOF AUC (base): 0.502265
OOF AUC (improved): 0.961737

======================================================================
BirdCLEF+ 2026 - Inference v5 (CPU Submission)
======================================================================

---

## Loading training artifacts...

Loaded arrays from: /kaggle/input/notebooks/blamerx/birdclef-2026-training/full_perch_arrays.npz
Keys in arrays: ['scores', 'emb', 'bc_indices', 'mapped_pos', 'unmapped_pos', 'mapped_bc_indices', 'selected_proxy_pos', 'idx_active_texture', 'idx_active_event', 'idx_mapped_active_texture', 'idx_mapped_active_event', 'idx_selected_proxy_active_texture', 'idx_selected_prioronly_active_texture', 'idx_selected_prioronly_active_event', 'idx_unmapped_inactive', 'proxy_map_keys', 'proxy_map_vals', 'cooccurrence_prob', 'pmi_matrix', 'config', 'labels', 'lr_probes_bytes', 'mlp_probes_bytes', 'emb_scaler_bytes', 'emb_pca_bytes', 'prior_tables_bytes']

Config: PCA=128, LR_C=0.7, Alpha=0.5
Temperature: 5.8471
Co-occurrence matrix: (234, 234)
PMI matrix: (234, 234)

Deserializing models...
LR probes: 60
Model expects 139 features

---

## Loading Perch v2 model (CPU)...

Perch model loaded from: /kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1

---

## Running Inference

No test files. Using 20 train soundscapes.

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1774155960.957197 66 device_compiler.h:196] Compiled cluster using XLA! This line is logged at most once for the lifetime of the process.

scores_raw: (240, 234), emb_test: (240, 1536)

Building predictions...
PCA embedding shape: (240, 128)
Base scores shape: (240, 234)
Base probs shape: (240, 234)
Base feature dim: 135
Using base + co-occurrence features (+4 dims)

Score range: -27.6767 to 19.2849

Creating submission...
Submission saved: /kaggle/working/submission.csv
Submission shape: (240, 235)
Prediction mean: 0.4410

======================================================================
Inference Complete!
======================================================================
Features: 139
Temperature: 5.8471
