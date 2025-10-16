# Gesture Translate

Optimized application for automatic translating from **(ASL)** to English Language

---

## 📖 Table of Contents

* [🚀 Key Features](#-key-features)
* [🌍 Dataset & Preprocessing](#-dataset--preprocessing)
* [⚙️ Installation](#️-installation)
* [🧪 Usage](#-usage)
* [🗺️ Roadmap](#️-roadmap)
* [📈 Methodology](#-methodology)
* [📊 Evaluation Metrics](#-evaluation-metrics)
* [👥 Team](#-team)
* [📜 License](#-license)

---

## 🚀 Key Features

* **Deep Neural Network (DNN)** for accurate sign classification
* **Complex gesture recognition** using hand landmarks & sequence modeling
* **Highly optimized** client-server architecture for real-time inference

---

## 🌍 Dataset & Preprocessing

We use the "Sign Language
Detection Using Images" dataset from [kaggle](https://www.kaggle.com/datasets/harshvardhan21/sign-language-detection-using-images).
This dataset was chosen for its large volume of curated image samples, which is crucial for effective deep learning training.

* Input: raw RGB frames from webcam
* Feature Extraction: using  for hand landmark detection
* Gesture Classification: deep model trained on sequences of landmark vectors

[▶️ View Preprocessing Code](ml/dataset.ipynb)

---

## ⚙️ Installation

1. Clone repository:

   ```bash
   git clone https://github.com/LowIQCoder/gesture_translate.git
   cd gesture_translate
   ```
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

For correct work you must set `KAGGLEHUB_CACHE`
```bash
export KAGGLEHUB_CACHE=./data/raw
```

If you want to look at model training process first of all get and preprocess data.

```bash
python -m data.preprocess
```

You will see 2 new folders

```
. 
└─── data
    └─── raw         # Here you will have dataset from kaggle
    └─── processed   # Here tou will have preprocessed features

```

### 🏋️‍♂️ Model Training

Now with ready data you can train model

To track model metrics we use [MLFlow](https://mlflow.org/). To do so, run tracking server with docker

```bash
docker compose up --build
```

With MLFlow running, launch training script

```bash
python -m src.ml.train
```

Now access MLFlow UI on http://localhost:5000 to see model training process 

### 🚗 App Usage

With trained model you can lauch our app

To do so simply run our application
```bash
python -m src.backend.app
```

And show somethig!

---

## 🗺️ Roadmap

**Planned timeline:**

* Week 1–2: Literature review, requirements, dataset collection
* Week 3: Build baseline MLP model
* Week 4–5: Implement and train LSTM model
* Week 6: Develop core application frontend
* Week 7: Integrate trained model with backend
* Week 8: Optimize model (quantization, pruning)
* Week 9: System testing and evaluation
* Week 10: Final report and release

---

## 📈 Methodology

**Tech Stack**

* **Python**— main development language
* **OpenCV**— image preprocessing
* **Mediapipe**— real-time hand landmark detection
* **PyTorch**— model training and inference

**Pipeline**

1. Capture video frames
2. Extract hand landmarks
3. Classify sequences with a deep learning model
4. Return predicted word as text/speech

---

## 📊 Evaluation Metrics

We will evaluate our model using:
**Performance**

* Top-1 and Top-5 Accuracy
* Precision, Recall, F1-score
* Confusion Matrix visualization

**Efficiency**

* Inference Time (latency) per frame
* Model size (MB)

**Target:** >90% accuracy with <33ms latency

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

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Innopolis — 2025

