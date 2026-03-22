# BirdCLEF+ 2026 - Complete Competition Guide

## 📊 COMPETITION OVERVIEW

### Task
- **Goal:** Identify wildlife species from audio recordings in Brazilian Pantanal
- **Species:** 234 classes (birds, amphibians, mammals, reptiles, insects)
- **Input:** 1-minute soundscape recordings (32kHz, OGG format)
- **Output:** Probability for each species for each 5-second window
- **Metric:** Macro-averaged ROC-AUC

### Data Structure
```
train_audio/          → Individual species recordings (~35,549 files)
train_soundscapes/    → Labeled soundscape recordings (~10,658 files)
test_soundscapes/     → ~600 one-minute recordings (hidden)
train.csv            → Metadata for training audio
taxonomy.csv         → 234 species info (class: Aves, Amphibia, etc.)
train_soundscapes_labels.csv → 5-second segment labels
sample_submission.csv → Format: row_id + 234 species probability columns
```

### Submission Format
```
row_id = [soundscape_filename]_[end_time]
Example: BC2026_Test_0001_S05_20250227_010002_20

Predict: Probability (0-1) for each of 234 species per 5-second segment
Each 1-minute audio → 12 predictions (at 5, 10, 15, ..., 60 seconds)
```

---

## 2. KEY CONSTRAINTS (CRITICAL!)

| Constraint | Value | Impact |
|------------|-------|--------|
| **Runtime** | 90 minutes (CPU) | Must optimize inference speed! |
| **GPU** | DISABLED (1 minute only) | Must use CPU-friendly models |
| **Internet** | Disabled | Can't download models at runtime |
| **Submission file** | submission.csv | Must match exact format |

### ⚠️ IMPLICATIONS:
1. **Pre-compute everything possible** - model weights, features, etc.
2. **Use efficient models** - EfficientNet-based (Perch)
3. **Batch processing** - process 600 test files efficiently
4. **~600 test files × 12 segments = 7200 predictions × 234 species**

---

## 3. WINNING APPROACH: Perch 2.0 + LR Probes

### What is Perch?
- **Google DeepMind's bioacoustics foundation model**
- Pre-trained on **~15,000 species** (birds + other taxa)
- Based on **EfficientNet-B3** architecture
- Achieves **SOTA on BirdSET and BEANS benchmarks**

### Perch 2.0 Key Features:
1. **Supervised pre-training** (not self-supervised!)
2. **Multi-taxa training** - birds, amphibians, mammals, insects
3. **Self-distillation** with prototype learning classifier
4. **Source prediction** auxiliary loss
5. **Multi-source mixup** augmentation

### Model Paths:
| Environment | Path |
|-------------|------|
| GPU Training | `/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2/2` |
| CPU Inference | `/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1` |

---

## 4. OUR SOLUTION EVOLUTION

### v1: Baseline (CPU)
- Proved Perch embeddings work
- Established two-notebook strategy
- No train_audio, no pseudo-labeling

### v2: Soundscapes-Only (LB: 0.907)
- PCA trained on soundscapes only
- LR probes only
- No train_audio, no pseudo-labeling
- **PCA Variance: ~79%**

### v3: Advanced Features (LB: 0.899 ❌)
- Added train_audio + pseudo-labeling
- MLP ensemble added
- **FAILED**: PCA variance collapsed to 39%
- MLP hurt performance (0.852 vs LR 0.918)
- Pseudo-labels added noise

### v4: Simplified & Effective (Expected LB: 0.915-0.925)
- **PCA trained on soundscapes only (fixed!)**
- **LR probes only (MLP removed)**
- **Sample weights (10x soundscapes)**
- **PCA dimensions: 128 (up from 64)**
- **PCA Variance: 89%** ✅
- **OOF AUC: 0.933** ✅

---

## 5. TRAINING PIPELINE (v4)

### Phase 1: Data Loading
```
1. Load taxonomy.csv, sample_submission.csv
2. Load train_soundscapes_labels.csv (handle duplicates!)
3. Load train.csv for train_audio metadata
4. Parse filename → site, hour_utc
5. Build label matrix (234 classes)
```

### Phase 2: Perch Embedding Extraction (GPU)
```
1. Load Perch v2 GPU model
2. Process train_soundscapes (739 labeled windows)
3. Process train_audio (35,549 clips)
4. Extract 1536-dim embeddings for each
5. Map Perch labels to competition labels via taxonomy
```

### Phase 3: Prior Tables
```
1. Compute global class probabilities
2. Compute site-specific probabilities
3. Compute hour-specific probabilities
4. Compute site-hour joint probabilities
5. Apply smoothing for rare combinations
```

### Phase 4: Domain-Aware PCA (v4 Key!)
```
1. Fit StandardScaler on SOUNSCAPE embeddings ONLY
2. Fit PCA on SOUNSCAPE embeddings ONLY
3. Transform train_audio embeddings using fitted scaler/PCA
4. This ensures PCA represents test distribution
```

### Phase 5: Sample Weighting (v4 New!)
```
sample_weights = {
    soundscape: 10.0,    # High weight - same domain as test
    train_audio: 1.0,    # Normal weight - different domain
}
```

### Phase 6: LR Probe Training
```
1. Build features: PCA + raw_score + prior + temporal
2. Train LogisticRegression per class (C=0.50, liblinear)
3. Use class_weight='balanced' + sample_weights
4. Only train on classes with >= 8 positive samples
```

### Phase 7: Save Artifacts
```
np.savez_compressed(
    "full_perch_arrays.npz",
    embeddings=soundscape_emb,
    lr_probes_bytes=pickle(lr_probes),
    emb_scaler_bytes=pickle(scaler),
    emb_pca_bytes=pickle(pca),
    prior_tables_bytes=pickle(priors),
    config=json.dumps({...}),
    ...
)
```

---

## 6. INFERENCE PIPELINE (v4)

### Step-by-Step
```python
# 1. Load artifacts (30 sec)
arrays = np.load("full_perch_arrays.npz")
lr_probes = pickle.loads(arrays['lr_probes_bytes'])
scaler = pickle.loads(arrays['emb_scaler_bytes'])
pca = pickle.loads(arrays['emb_pca_bytes'])

# 2. Load Perch CPU (30 sec)
perch = tf.saved_model.load(PERCH_CPU)
infer_fn = perch.signatures["serving_default"]

# 3. Load test audio (~5 min)
with ThreadPoolExecutor(8) as ex:
    audio = np.stack(list(ex.map(read_audio, test_paths)))

# 4. Run Perch (~60 min)
for batch in batches:
    outputs = infer_fn(inputs=batch)
    embeddings.append(outputs["embedding"])

# 5. Transform embeddings
Z = pca.transform(scaler.transform(embeddings))

# 6. Apply LR probes (~2 min)
for cls_idx, clf in lr_probes.items():
    pred = clf.decision_function(features)
    scores[:, cls_idx] = (1-alpha)*base + alpha*pred

# 7. Create submission (~1 min)
submission = pd.DataFrame({'row_id': row_ids})
for i, species in enumerate(PRIMARY_LABELS):
    submission[species] = sigmoid(scores[:, i])
submission.to_csv('submission.csv', index=False)
```

---

## 7. KEY LEARNINGS

### What Worked ✅
1. **Soundscapes-only PCA** - Captures test distribution
2. **LR Probes** - Simple but effective
3. **Sample Weighting** - Safe use of train_audio
4. **Higher PCA Dims (128)** - Better representation

### What Failed ❌
1. **Mixed PCA** - train_audio dilutes representation
2. **MLP Ensemble** - Overfits to wrong distribution
3. **Aggressive Pseudo-labeling** - Adds noise
4. **OOF with Pseudo-labels** - Misleading metric

### Domain Mismatch Issue
```
train_audio:       Clean recordings, focused on single species
train_soundscapes: Field recordings, multiple species, noise
test_soundscapes:  Same environment as train_soundscapes

→ Solution: Train PCA on soundscapes only, weight soundscapes 10x
```

---

## 8. v5 IMPROVEMENT IDEAS

### High Priority
1. **Test-Time Augmentation (TTA)**
   - Time stretching (0.95x, 1.0x, 1.05x)
   - Pitch shifting (±1 semitone)
   - Expected: +1-2% LB

2. **Multi-PCA Ensemble**
   - Train with 64, 96, 128 dims
   - Average predictions
   - Expected: +0.5-1% LB

3. **Conservative Pseudo-labeling**
   - Threshold 0.98+ (not 0.95)
   - Per-class thresholds
   - Expected: +0.5-1% LB

### Medium Priority
4. **Asymmetric Loss** - Different weights for pos/neg
5. **Per-Class Weight Optimization** - Rare species higher weight
6. **Enhanced Temporal Smoothing** - Adaptive by class type

### Research Sources
- [BirdCLEF 2024 1st Place](https://www.kaggle.com/competitions/birdclef-2024/writeups/team-kefir-1st-place-solution)
- [BirdCLEF 2025 1st Place](https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n)
- [Perch 2.0 Paper](https://arxiv.org/html/2508.04665v1)
- [Asymmetric Loss Paper](https://arxiv.org/abs/2009.14119)

---

## 9. FILES REFERENCE

### Training Scripts
```
birdclef_2026_training_v4.py  → Current training script
  - USE_TRAIN_AUDIO = True
  - USE_PSEUDO_LABELING = False
  - PCA_DIM = 128
  - SAMPLE_WEIGHTS = {soundscape: 10.0, train_audio: 1.0}
```

### Inference Scripts
```
birdclef_2026_inference_v4.py  → Current inference script
  - BATCH_FILES = 50 (safe for 30 GiB RAM)
  - Uses single NPZ file for all artifacts
```

### Output Files
```
full_perch_arrays.npz  → All training artifacts
full_perch_meta.parquet → Metadata (row_id, site, hour)
```

---

## 10. METRICS SUMMARY

| Version | OOF AUC | PCA Var | LB Score | Status |
|---------|---------|---------|----------|--------|
| v1 | - | - | - | Baseline |
| v2 | ~0.918 | ~79% | **0.907** | ✅ Best |
| v3 | 0.920 | 39% | 0.899 | ❌ Failed |
| v4 | **0.933** | **89%** | **?** | ⏳ Pending |

---

**Last Updated**: 2026-03-21
**Current Best LB**: 0.907 (v2)
**v4 LB Score**: Pending (submission running)
