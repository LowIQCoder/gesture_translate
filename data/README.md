Here's an improved version of your README with better organization, clarity, and professional formatting:

# Dataset Directory

This directory contains all data-related assets for the project, organized following data science best practices.

## 📁 Directory Structure

```
data/
├── raw/                      # Original, immutable raw data
│   └── data/                 # Folder with images
│       ├── 1
│       ├── 2
│       ├── ...
│       └── Z
│
├── processed/                    # Cleaned and processed data
│   ├── train.parquet             # Training split
│   ├── val.parquet               # Validation split
│   └── test.parquet              # Test split
│
├── preview/                      # Data exploration and visualization
│   ├── preview.ipynb             # Jupyter notebook for data preview
│   └── *.jpg                     # Data visualization samples
│
├──  models/                      # Model artifacts and configurations
│   ├── best_model.pth            # Best model checkpoint
│   ├── model_configuration.json  # Best model configuration files
│   └── model_summary.txt         # Model architecture summary
│
├── preprocessing.py             # Data preprocessing scripts
└── features.py                  # Feature engineering utilities
```

## 🔧 Scripts

### `preprocessing.py`
Contains data cleaning, transformation, and preparation pipelines.

To run use
```bash
python -m data.preprocessing # Automatically downloads and preprocess dataset
```

Or specify path to downloaded dataset
In case you have dataset downloaded
```bash
python -m data.preprocessing ./data/raw/data # Note that data contains folders 1, 2, 3 ...
```

### `features.py`
Includes feature engineering utilities and data transformation functions.

Each feature vector is a **(,84)** dim vector, with stacked **x** and **y** coordinates of both hands. All features are normalized.
