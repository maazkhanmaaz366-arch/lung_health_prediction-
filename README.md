# Lung Cancer Risk Predictor

A machine-learning project with a desktop interface for exploring lung-cancer risk predictions from a synthetic dataset. It includes a TensorFlow model training script, a Tkinter application that loads the saved model, and a separate logistic-regression experiment.

> **Medical disclaimer:** This project is for educational use only. Its predictions are not a diagnosis, screening result, or substitute for advice from a qualified healthcare professional. The dataset is synthetic, and the model has not been validated for clinical use.

## Project contents

```text
.
├── UI.py                              # Tkinter prediction application
├── lungsPrediction.py                 # TensorFlow training and evaluation
├── lcs_synthetic_20000.csv            # Main 20,000-row dataset
├── logisticRegresion/
│   ├── final_lab.py                   # Logistic-regression experiment
│   └── lcs_synthetic_20000.csv        # Dataset copy for that experiment
├── model/
│   ├── best_model.keras               # Best checkpoint from training
│   ├── final_model.keras              # Model loaded by the UI
│   ├── feature_names.pkl              # Model input feature order
│   ├── optimal_threshold.pkl          # Selected training threshold
│   └── scaler.pkl                     # Fitted StandardScaler
└── results/
    └── <run_timestamp>/
        ├── classification_report.csv
        ├── confusion_matrix.csv
        └── training_history.png
```

The training script reads the root-level CSV and writes model artifacts to `model/` and timestamped evaluation outputs to `results/`. Re-running training replaces the shared model artifacts but creates a new results directory.

## Dataset

The CSV contains 20,000 synthetic records with 15 input columns and one target column:

- `GENDER`, `AGE`
- `SMOKING`, `YELLOW_FINGERS`, `ANXIETY`, `PEER_PRESSURE`, `CHRONIC_DISEASE`
- `FATIGUE`, `ALLERGY`, `WHEEZING`, `ALCOHOL_CONSUMING`, `COUGHING`
- `SHORTNESS_OF_BREATH`, `SWALLOWING_DIFFICULTY`, `CHEST_PAIN`
- `LUNG_CANCER` (`YES` or `NO`)

The symptom columns are represented as `1` (no) and `2` (yes) in the CSV. The TensorFlow training script maps them to `0` and `1`; gender is encoded as `M=0`, `F=1`, and the target as `NO=0`, `YES=1`. Features are standardized using a scaler fitted on the training partition.

## Setup

Use a Python environment with TensorFlow available for the installed Python version. From the project root, for example:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn tensorflow matplotlib joblib seaborn
```

Tkinter is part of many standard Python distributions; if it is not included in your Python installation, install or enable the Tk support provided by that distribution.

## Run the desktop application

The app loads the artifacts in `model/` relative to `UI.py`. Keep those files in place, then run:

```powershell
python UI.py
```

Enter an age from 1 to 120, select a gender and health indicators, then choose the prediction action. The app displays a model score as a percentage and groups it into low, moderate, or high bands. These UI bands are presentation categories, not clinically established risk thresholds.

## Train and evaluate the TensorFlow model

From the project root:

```powershell
python lungsPrediction.py
```

The script uses stratified train/validation/test partitions, trains a feed-forward neural network with early stopping and class weighting, evaluates candidate probability thresholds, and saves:

- `best_model.keras` and `final_model.keras`
- `scaler.pkl`, `feature_names.pkl`, and `optimal_threshold.pkl`
- A classification report, confusion matrix, and training-loss plot in a timestamped `results/` subdirectory

The saved `optimal_threshold.pkl` records the threshold selected during training. The current UI displays the model's raw prediction score and uses fixed 30% and 60% boundaries for its labels.

## Logistic-regression experiment

The separate analysis script performs preprocessing, plots an age histogram and a correlation heatmap, then trains and evaluates a scikit-learn logistic-regression model:

```powershell
python logisticRegresion/final_lab.py
```

This experiment uses the CSV copy inside `logisticRegresion/` and prints its evaluation metrics to the console.

## Existing evaluation snapshots

The checked-in `results/` folders contain prior run outputs. The highest accuracy among the available classification reports is **87.275%** in `results/20251208_125740/`. This is a historical result for that saved run, not a guarantee of current or clinical performance.
