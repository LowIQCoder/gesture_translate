Here's an improved version of your README with better organization, clarity, and professional formatting:

# Dataset Directory

This directory contains all data-related assets for the project, organized following data science best practices.

## 📁 Directory Structure

```
data/
├── raw/                      # Original, immutable raw data
│   ├── annotations.csv       # Labels and annotations
│   └── solvo_mediapipe.json  # Raw dataset from MediaPipe
│
├── processed/                # Cleaned and processed data
│   ├── train.parquet         # Training split
│   ├── val.parquet           # Validation split
│   └── test.parquet          # Test split
│
├── preview/                 # Data exploration and visualization
│   ├── preview.ipynb         # Jupyter notebook for data preview
│   └── *.gif                 # Data visualization samples
│
├──  models/                  # Model artifacts and configurations
│   ├── best_model.pth        # Best model checkpoint
│   ├── model_configuration/  # Best model configuration files
│   └── model_summary.txt     # Model architecture summary
│
├── preprocessing.py          # Data preprocessing scripts
└── features.py               # Feature engineering utilities
```

## 🗂️ Contents Description

### Raw Data (`raw/`)
- **Purpose**: Original source data - do not modify directly
- **Files**:
  - `annotations.csv`: Contains all labels and annotations
  - `solvo_mediapipe.json`: Raw MediaPipe output data

### Processed Data (`processed/`)
- **Purpose**: Cleaned, split, and ready-to-use datasets
- **Format**: Parquet files for efficient storage
- **Splits**:
  - Training set (`train.parquet`)
  - Validation set (`val.parquet`) 
  - Test set (`test.parquet`)

### Preview (`preview/`)
- **Purpose**: Data exploration and visualization
- **Files**:
  - `preview.ipynb`: Interactive notebook for data analysis
  - GIF files: Visual representations of the dataset

### Models (`models/`)
- **Purpose**: Trained model artifacts and configurations
- **Files**:
  - `best_model.pth`: PyTorch model checkpoint
  - `model_configuration/`: Model hyperparameters and settings
  - `model_summary.txt`: Architecture overview

## 🔧 Scripts

### `preprocessing.py`
Contains data cleaning, transformation, and preparation pipelines.

### `features.py`
Includes feature engineering utilities and data transformation functions.
