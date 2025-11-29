# Gesture Translate

Optimized application for automatic translating from **(ASL)** to English Language

---

## 📖 Table of Contents

* [🌍 Dataset & Preprocessing](#-dataset--preprocessing)
* [⚙️ Installation](#️-installation)
* [🧪 Usage](#-usage)
* [📈 Methodology](#-methodology)
* [👥 Team](#-team)
* [📜 License](#-license)

---

## 🌍 Dataset & Preprocessing

We use the "Sign Language
Detection Using Images" dataset from [kaggle](https://www.kaggle.com/datasets/harshvardhan21/sign-language-detection-using-images).

* Input: raw RGB frames from webcam
* Feature Extraction: using  for hand landmark detection
* Gesture Classification: deep model trained on landmark vectors

---

## ⚙️ Installation

1. Clone repository:

   ```bash
   git clone https://github.com/LowIQCoder/gesture_translate.git
   cd gesture_translate
   ```
   **NOTE that path to repository must NOT contain any CYRILLIC LETTERS**
2. Create and activate **Python 3.12** virtual environment:

   ```bash
   python -m venv .venv
   .venv/Scripts/activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🧪 Usage

### 📝 Data Preprocessing
If you want to look at model training process first of all get and preprocess data. If you do ton have dataset just run
```bash
python -m data.preprocessing # Automatically downloads and preprocess dataset
```

**NOTE that by default on Windows system dataset will be installed at <USER_HOME>\.cache\kagglehub\datasets\. In case if path contains CYRILLIC LETTERS OpenCV will fail to open images**

In case you have dataset downloaded
```bash
python -m data.preprocessing ./data/raw/data # Note that data contains folders 1, 2, 3 ...
```

After you will see 2 new folders

```
. 
data/
├── raw/                      # Original, immutable raw data
│   └── data/                 # Folder with images
│       ├── 1
│       ├── 2
│       ├── ...
│       └── Z
│
├── processed/                # Cleaned and processed data
│   ├── train.parquet         # Training split
│   ├── val.parquet           # Validation split
│   └── test.parquet          # Test split

```

### 🏋️‍♂️ Model Training

Now with ready data you can train model

To track model metrics we use [MLFlow](https://mlflow.org/). To do so, run tracking server with docker

```bash
docker compose up mlflow --build
```

With MLFlow running, launch training script

```bash
python -m src.ml.train
```

Now access MLFlow UI on http://localhost:5000 to see model training process 

### 🚗 App Usage

With trained model you can lauch our frontend and backend

```bash
docker compose up --build
```

Now navigate to http://localhost:8080 and show gesture!

---

## 📈 Methodology

**Tech Stack**

* **Python** - main development language
* **OpenCV** - image preprocessing
* **Mediapipe** - real-time hand landmark detection
* **PyTorch** - model training
* **Flask** - API server
* **ONNX** - model inference
---

## 👥 Team

| Member | Role           | Contribution                                        |
| ------ | -------------- | --------------------------------------------------- |
| Marsel Berheev       | ML Engineer    | Model architecture, training, evaluation            |
| Vlad Strelkov       | ML Engineer    | Data preprocessing, pipeline, MediaPipe integration |
| Ekaterina Petrova       | MLOps Engineer | App development, deployment, cloud infrastructure   |

**Contacts**

* [m.berheev@innopolis.university](mailto:m.berheev@innopolis.university)
* [vl.strelkov@innopolis.university](mailto:vl.strelkov@innopolis.university)
* [ek.petrova@innopolis.university](mailto:ek.petrova@innopolis.university)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Innopolis - 2025
