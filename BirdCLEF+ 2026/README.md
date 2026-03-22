# BirdCLEF+ 2026 Kaggle Competition

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-2026-blue.svg)](https://www.kaggle.com/competitions/birdclef-2026/overview)

![Kaggle Competition Header](https://www.kaggle.com/competitions/129329/images/header)

This repository contains the code for the [BirdCLEF+ 2026 Kaggle Competition](https://www.kaggle.com/competitions/birdclef-2026/overview). The goal of this competition is to classify bird sounds to support biodiversity monitoring in the Pantanal wetlands. This solution uses a pre-trained Perch v2 model and trains linear probes for bird sound classification.

## Competition Details

### Description

How do you protect an ecosystem you can’t fully see? One way is to listen.

This competition involves building models that automatically identify wildlife species from their vocalizations in audio recordings collected across the Pantanal wetlands. This work will support more reliable biodiversity monitoring in one of the world’s most diverse and threatened ecosystems.

Understanding how ecological communities respond to environmental change and restoration efforts is a central challenge in conservation science. The Pantanal — a wetland spanning 150,000+ km² across Brazil and neighboring countries — is home to over 650 bird species plus countless other animals, yet much of it remains unmonitored. Seasonal flooding, wildfires, agricultural expansion, and climate change make regular fieldwork challenging.

### Goal of the Competition

Conventional biodiversity monitoring across vast, remote regions is expensive and logistically demanding. To help address these challenges, a growing network of 1,000 acoustic recorders is being deployed across the Pantanal, running continuously to capture wildlife sounds across different habitats and seasons. Continuous audio recording allows researchers to capture multi-species soundscapes over extended periods, providing a community-level perspective on biodiversity dynamics. But the sheer volume of audio is too large to review manually, and labeled species data is limited.

This competition focuses on the development of machine learning models that identify wildlife species from passive acoustic monitoring (PAM). Proposed approaches should work across different habitats, withstand the constraints of messy, field-collected data, and support evidence-based conservation decisions. Successful solutions will help advance biodiversity monitoring in the last wild places on Earth, including research initiatives in the Pantanal wetlands of Brazil.

Listening carefully, and at scale, may be one of the most effective tools available to protect this landscape.

## Evaluation

The evaluation metric for this contest is a version of **macro-averaged ROC-AUC** that skips classes that have no true positive labels.

## Project Structure

- `birdclef_2026_training.py`: This script handles the training of the model. It loads the data, prepares it, and then trains the classification model.
- `birdclef_2026_inference.py`: This script is used for making predictions on the test data. It loads the trained model and outputs a submission file.
- `LICENSE`: The license for this project.
- `README.md`: This file.

## Getting Started

### Prerequisites

- Python 3
- TensorFlow 2.20
- Pandas
- NumPy
- scikit-learn
- SoundFile
- joblib
- tqdm

### Training

To train the model, run the `birdclef_2026_training.py` script.

```bash
python birdclef_2026_training.py
```

This will train the model and save the necessary artifacts. The script performs the following steps:
1.  **Installs Dependencies**: Installs the required version of TensorFlow.
2.  **Loads Data**: Loads the taxonomy, sample submission, and soundscape labels.
3.  **Parses Labels**: Cleans and processes the labels.
4.  **Perch Model Mapping**: Loads the pre-trained Perch v2 model and maps the labels.
5.  **Frog Proxies**: Builds frog proxies for unmapped classes.
6.  **Perch Inference**: Runs inference on the training soundscapes to get embeddings and raw scores.
7.  **Builds OOF**: Creates out-of-fold predictions.
8.  **Trains Probes**: Trains linear probes on top of the Perch embeddings.
9.  **Saves Artifacts**: Saves the trained models, scalers, and other artifacts.

### Inference

To run inference, use the `birdclef_2026_inference.py` script.

```bash
python birdclef_2026_inference.py
```

This will load the trained model and generate a `submission.csv` file in the root directory. The script performs the following steps:
1.  **Installs Dependencies**: Installs the required version of TensorFlow.
2.  **Loads Artifacts**: Loads all the training artifacts.
3.  **Loads Perch Model**: Loads the pre-trained Perch v2 model.
4.  **Runs Inference**: Runs inference on the test soundscapes.
5.  **Builds Predictions**: Fuses scores with priors and applies the trained probes.
6.  **Creates Submission**: Generates the final `submission.csv` file.

## Code Requirements

This is a Code Competition. Submissions must be made through Notebooks.
- CPU Notebook &lt;= 90 minutes run-time
- GPU Notebook submissions are disabled.
- Internet access disabled.
- Freely & publicly available external data is allowed, including pre-trained models.
- Submission file must be named `submission.csv`.

## Model

This solution uses a hybrid approach:
-   **Perch v2 Model**: A pre-trained model from Google for bird vocalization classification, used to generate embeddings and initial predictions.
-   **Linear Probes**: `LogisticRegression` models from scikit-learn are trained on top of the Perch embeddings and other features to refine the predictions for the competition-specific classes.

## Dataset Description

Your challenge in this competition is to identify which species (birds, amphibians, mammals, reptiles, insects) are calling in recordings made in the Brazilian Pantanal. This is an important task for scientists who monitor animal populations for conservation purposes. More accurate solutions could enable more comprehensive monitoring. This competition uses a hidden test set. When your submitted notebook is scored, the actual test data will be made available to your notebook.

### Files

-   **train_audio/**: The training data consists of short recordings of individual bird, amphibian, reptile, mammal, and insect sounds generously uploaded by users of xeno-canto.org and iNaturalist. These files have been resampled to 32 kHz where applicable to match the test set audio and converted to the ogg format.
-   **test_soundscapes/**: When you submit a notebook, the `test_soundscapes` directory will be populated with approximately 600 recordings to be used for scoring. They are 1 minute long and in ogg audio format, resampled to 32 kHz.
-   **train_soundscapes/**: Additional audio data from roughly the same recording locations as the `test_soundscapes`. Some of the train_soundscapes have been labeled by expert annotators, and we provide the ground truth for a subset of `train_soundscapes` in `train_soundscapes_labels.csv`.
-   **train.csv**: A wide range of metadata is provided for the training data, including `primary_label`, `secondary_labels`, `latitude`, `longitude`, `author`, `filename`, `rating`, and `collection`.
-   **sample_submission.csv**: A valid sample submission file.
-   **taxonomy.csv**: Data on the different species, including iNaturalist taxon ID and class name. The 234 rows of this file represent the 234 class columns in the submission file.
-   **recording_location.txt**: Some high-level information on the recording location (Pantanal, Brazil).

## License

This project is licensed under the terms of the `LICENSE` file.
