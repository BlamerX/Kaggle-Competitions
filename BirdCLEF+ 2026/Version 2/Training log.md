# LB Score is 0.907

============================================================
BirdCLEF+ 2026 - Training (ALL labeled windows)
============================================================

Loading competition data...
Classes: 234, Raw label rows: 739
After dedup: 739 labeled windows from 66 files
Label density: 1.81%

Loading Perch model...
Mapped: 203, Unmapped: 31

Building frog proxies...
Texture: 42, Event: 33, Proxies: 3

------------------------------------------------------------
Running Perch on ALL labeled training soundscapes...
------------------------------------------------------------

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1774005831.642186      66 device_compiler.h:196] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.

Total windows processed: 792
scores_raw: (792, 234), emb_train: (792, 1536)
Y_TRAIN: (792, 234), positive rate: 1.68%

Building OOF...

OOF AUC: 0.502265

Training probes...
PCA: 64, var: 0.7934

Trained 53 LR probes, 53 MLP probes

Evaluating OOF with probes...

OOF AUC (base): 0.502265
OOF AUC (final): 0.883714

Saving artifacts...

============================================================
Training Complete!
============================================================

Output Files:
  - full_perch_arrays.npz    → embeddings + scores + models + indices
  - full_perch_meta.parquet  → metadata

Metrics:
  OOF AUC (base):  0.502265
  OOF AUC (final): 0.883714




============================================================
BirdCLEF+ 2026 - Inference
============================================================

Loading artifacts...
Classes: 234, LR probes: 53, MLP probes: 53

Loading Perch model...
Perch loaded

------------------------------------------------------------
Running Inference
------------------------------------------------------------
No test files. Using 20 train soundscapes.

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1774006824.213585      66 device_compiler.h:196] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.

scores_raw: (240, 234), emb_test: (240, 1536)

Building predictions...

Score range: -19.2810 to 14.9108

============================================================
Inference Complete!
============================================================
Submission shape: (240, 235)
                                     row_id   1161364    116570   1176823  \
0   BC2026_Train_0001_S08_20250606_030007_5  0.436655  0.464281  0.200763   
1  BC2026_Train_0001_S08_20250606_030007_10  0.516245  0.468166  0.160440   
2  BC2026_Train_0001_S08_20250606_030007_15  0.492779  0.300602  0.177227   

    1491113   1595929    209233     22930  
0  0.000033  0.000335  0.721654  0.149820  
1  0.000029  0.000335  0.741231  0.121439  
2  0.000043  0.000335  0.702286  0.133897  