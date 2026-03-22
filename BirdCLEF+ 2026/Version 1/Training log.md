# LB Score is 0.904

============================================================
BirdCLEF+ 2026 - Training
============================================================

Loading data...
Classes: 234
Full files: 59, Windows: 708

Loading Perch model...
Mapped: 203, Unmapped: 31

Building frog proxies...
Frog proxies: 3

Running Perch on training soundscapes...

Perch: 100%
 4/4 [07:36<00:00, 108.30s/it]

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1773958676.279776      74 device_compiler.h:196] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.

scores_raw: (708, 234), emb_full: (708, 1536)

Building OOF...

OOF: 100%
 5/5 [00:00<00:00, 90.63it/s]

OOF AUC: 0.485057

Training probes...
PCA: 64, var: 0.8147

Probes: 100%
 52/52 [00:00<00:00, 80.20it/s]

Trained 52 probes

Saving artifacts...

============================================================
Training Complete!
============================================================
OOF AUC: 0.485057
Probes: 52



============================================================
BirdCLEF+ 2026 - Inference
============================================================

Loading artifacts...
Classes: 234, Probes: 52

Loading Perch model...
Perch loaded

------------------------------------------------------------
Running Inference
------------------------------------------------------------
No test files. Using 20 train soundscapes.

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1773959741.551252      65 device_compiler.h:196] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.

scores_raw: (240, 234), emb_test: (240, 1536)

Building predictions...

Score range: -28.3123 to 23.0142

============================================================
Inference Complete!
============================================================
Submission shape: (240, 235)
                                     row_id   1161364    116570   1176823  \
0   BC2026_Train_0001_S08_20250606_030007_5  0.436655  0.637402  0.200763   
1  BC2026_Train_0001_S08_20250606_030007_10  0.516245  0.679265  0.160440   
2  BC2026_Train_0001_S08_20250606_030007_15  0.492779  0.387460  0.177227   

    1491113   1595929    209233     22930  
0  0.000011  0.000335  0.721654  0.149820  
1  0.000008  0.000335  0.741231  0.121439  
2  0.000014  0.000335  0.702286  0.133897  