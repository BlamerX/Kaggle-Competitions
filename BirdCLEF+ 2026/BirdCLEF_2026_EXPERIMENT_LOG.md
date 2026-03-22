# BirdCLEF+ 2026 - Experiment Log

## Overview

This document tracks all experiments, their results, and lessons learned for the BirdCLEF+ 2026 competition.

---

## Competition Summary

| Item | Value |
|------|-------|
| **Task** | Multi-label species classification from audio |
| **Classes** | 234 species (birds, amphibians, mammals, reptiles, insects) |
| **Input** | 1-minute soundscape recordings (32kHz) |
| **Output** | Probability for each species for each 5-second window |
| **Metric** | Macro-averaged ROC-AUC |
| **Constraint** | 90 minutes CPU inference time |

---

## Model Architecture

### Base Model: Google Perch v2
- **Architecture**: EfficientNet-B3 based
- **Embedding Dimension**: 1536
- **Pre-training**: Multi-taxa (birds, amphibians, mammals, insects)
- **Training**: Supervised with self-distillation

### Classifier: Linear Probes
- **PCA**: Dimensionality reduction (64-128 dims)
- **Logistic Regression**: Per-class binary classifiers
- **Features**: PCA embedding + raw score + prior + temporal features

---

## Experiment History

### Version 1: Baseline (Not Tracked)
- **Status**: Initial exploration
- **Notes**: Used basic Perch embeddings with simple classifier
- **Result**: Established baseline approach

---

### Version 2: Soundscapes-Only Training
**Date**: Initial submission

| Component | Setting |
|-----------|---------|
| PCA Training Data | Soundscapes only |
| PCA Dimensions | 64 |
| Classifier | LR only |
| Pseudo-labeling | No |
| train_audio | No |
| Sample Weights | No |

**Results**:
| Metric | Value |
|--------|-------|
| OOF AUC | ~0.918 |
| LB Score | **0.907** |
| PCA Variance | ~79% |

**Key Learnings**:
- ✅ Soundscapes-only PCA captures test distribution well
- ✅ LR probes work well for this task
- ✅ Simplicity wins - no complex ensemble needed

---

### Version 3: Advanced Features (FAILED)
**Date**: Second submission

| Component | Setting |
|-----------|---------|
| PCA Training Data | Soundscapes + train_audio (MIXED) |
| PCA Dimensions | 64 |
| Classifier | LR + MLP ensemble (min reduction) |
| Pseudo-labeling | Yes (13,558 windows, threshold=0.95) |
| train_audio | Yes (35,549 clips) |
| Sample Weights | No |

**Results**:
| Metric | Value |
|--------|-------|
| OOF AUC | 0.920 (misleading!) |
| LB Score | **0.899** ❌ (WORSE than v2) |
| PCA Variance | 39% (COLLAPSED!) |

**Failure Analysis**:
| Issue | Root Cause | Impact |
|-------|-----------|--------|
| PCA Variance Collapse | Mixed train_audio + soundscape data | 79% → 39% variance |
| MLP Underperforming | LR=0.918, MLP=0.852 OOF | Dragged down ensemble |
| Pseudo-label Noise | ~5% noise rate at 0.95 threshold | Contaminated training |
| Misleading OOF | Pseudo-labels reinforced model bias | OOF looked good, LB failed |

**Key Learnings**:
- ❌ More data ≠ Better when domains differ
- ❌ train_audio has different acoustic characteristics than soundscapes
- ❌ MLP can overfit to wrong distribution
- ❌ Pseudo-labeling adds noise if not carefully controlled
- ❌ OOF can be misleading when pseudo-labels contaminate evaluation

---

### Version 4: Simplified & Effective
**Date**: Third submission (pending)

| Component | Setting |
|-----------|---------|
| PCA Training Data | Soundscapes only (FIXED) |
| PCA Dimensions | 128 (increased) |
| Classifier | LR only (MLP removed) |
| Pseudo-labeling | No (disabled) |
| train_audio | Yes (35,549 clips, weighted) |
| Sample Weights | Soundscapes=10x, train_audio=1x |

**Results**:
| Metric | Value |
|--------|-------|
| OOF AUC | **0.933** |
| PCA Variance | **0.8936** (excellent!) |
| LB Score | *Pending submission* |

**Expected LB**: 0.915-0.925 (based on OOF improvement and PCA variance)

**Key Improvements**:
- ✅ PCA variance restored (89% > v2's 79%)
- ✅ Removed MLP (was hurting performance)
- ✅ Removed pseudo-labeling (too noisy)
- ✅ Sample weights prioritize soundscape data
- ✅ Higher PCA dimensions (128 vs 64)

**Key Learnings**:
- ✅ Train PCA on soundscape data only (same domain as test)
- ✅ Sample weighting allows safe use of train_audio
- ✅ Simpler model (LR only) can outperform complex ensemble

---

## Version Comparison

| Feature | v2 | v3 | v4 |
|---------|-----|-----|-----|
| **LB Score** | 0.907 | 0.899 ❌ | *~0.92* |
| **OOF AUC** | ~0.918 | 0.920 | 0.933 |
| **PCA Variance** | ~79% | 39% ❌ | 89% ✅ |
| **PCA Data** | Soundscape | Mixed | Soundscape |
| **PCA Dims** | 64 | 64 | 128 |
| **Classifier** | LR | LR+MLP | LR |
| **Pseudo-label** | No | Yes | No |
| **train_audio** | No | Yes | Yes (weighted) |
| **Sample Weights** | No | No | Yes |

---

## Key Findings

### What Worked
1. **Soundscapes-only PCA**: Training PCA on soundscape embeddings captures test distribution
2. **LR Probes**: Simple logistic regression outperforms complex MLP for this task
3. **Sample Weighting**: 10x weight for soundscapes allows safe use of train_audio
4. **Higher PCA Dims**: 128 dimensions captures more information than 64

### What Failed
1. **Mixed PCA**: train_audio has different acoustic characteristics, dilutes PCA
2. **MLP Ensemble**: Overfits to train_audio distribution, hurts test performance
3. **Aggressive Pseudo-labeling**: Even at 0.95 threshold, adds noise
4. **OOF with Pseudo-labels**: Creates self-reinforcing bias

### Domain Mismatch Issue
```
train_audio:    Clean recordings, focused on single species
train_soundscapes: Field recordings, multiple species, background noise
test_soundscapes: Same environment as train_soundscapes
```
**Implication**: train_audio data is valuable but needs careful handling (weighting, separate PCA transform)

---

## Research Findings for Future Versions

### From BirdCLEF 2024 2nd Place Solution
1. **Iterative Pseudo-labeling**: 3 cycles of pseudo-label generation
2. **Domain Adaptation via Mixing**: Mix training samples with test domain samples
3. **Knowledge Distillation**: Transfer knowledge from ensemble to smaller model
4. **Checkpoint Soup**: Average weights from multiple epochs instead of early stopping

### From Research Papers
1. **Test-Time Augmentation (TTA)**: Average predictions over augmented inputs
2. **BirdAVES**: Alternative self-supervised model to Perch
3. **AudioProtoPNet**: Interpretable prototype-based classification

### Potential v5 Improvements
1. **Conservative Pseudo-labeling**: Higher threshold (0.98+) or use only for specific classes
2. **Test-Time Augmentation**: Time stretching, pitch shifting
3. **Multi-PCA Ensemble**: Train multiple models with different PCA dimensions
4. **Better Weighting Strategies**: Per-class weights based on class difficulty
5. **Domain Adaptation**: CORAL, MMD loss for aligning distributions

---

## Files Structure

```
/home/z/my-project/download/
├── birdclef_2026_training_v3.py    # Failed v3 training
├── birdclef_2026_training_v4.py    # Current v4 training
├── birdclef_2026_inference_v3.py   # v3 inference
├── birdclef_2026_inference_v4.py   # Current v4 inference
├── BirdCLEF_2026_EXPERIMENT_LOG.md # This file
├── BirdCLEF_2026_FILE_STRUCTURE_REFERENCE.md
├── BirdCLEF_2026_CPU_Strategy.md
└── BirdCLEF_2026_Complete_Guide.md
```

---

## Next Steps for v5

### High Priority
1. [ ] Submit v4 and verify LB score improvement
2. [ ] Implement conservative pseudo-labeling (threshold=0.98)
3. [ ] Test multi-PCA ensemble (64, 96, 128 dims)

### Medium Priority
1. [ ] Implement TTA (time stretching, pitch shifting)
2. [ ] Try BirdAVES as alternative embedding model
3. [ ] Experiment with CORAL domain adaptation

### Low Priority
1. [ ] Explore iterative pseudo-labeling (if v5 shows promise)
2. [ ] Checkpoint soup for model averaging
3. [ ] Per-class weight optimization

---

**Last Updated**: 2026-03-21
**Current Best LB**: 0.907 (v2)
**Expected v4 LB**: 0.915-0.925
