import kagglehub
from os import PathLike
import os

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd

import cv2
import mediapipe as mp

def get_kaggle_dataset(
        path: str | PathLike
    ) -> str:
    """Downloads dataset from KaggleHub
    
    Args:
        path (str, PathLike): Path to download to

    Returns:
        str: Path datased downloaded to
    """
    os.environ["KAGGLEHUB_CACHE"] = os.path.abspath(path)
    dataset_path = kagglehub.dataset_download("harshvardhan21/sign-language-detection-using-images")
    print(f"Dataset downloaded to: {os.path.abspath(dataset_path)}")
    return os.path.abspath(dataset_path)


def landmarks_to_features(detection_results):
    """Generate a feature vector from both hands.

    Args:
        detection_results: The results object from Mediapipe (hands.process()).

    Returns:
        np.ndarray: The feature vector of size (84,) where:
                    - First 42 values are right hand (x,y)
                    - Last 42 values are left hand (x,y)
                    - If a hand is missing, zeros are used.
    """
    features = np.zeros(84, dtype=np.float32)

    if not detection_results.multi_hand_landmarks:
        return features

    for hand_landmarks, hand_handedness in zip(
        detection_results.multi_hand_landmarks,
        detection_results.multi_handedness
    ):
        coords = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32).flatten()

        if hand_handedness.classification[0].label == "Right":
            features[0:42] = coords
        else:
            features[42:84] = coords

    return features

def preprocess_image(image_path, debug=False):
    """Preprocess the image for hand tracking and return landmark features.

    Args:
        image_path (str): The path to the image file.
        debug (bool, optional): Whether to display debug info. Defaults to False.

    Raises:
        ValueError: If the image cannot be loaded.

    Returns:
        np.ndarray: The processed feature vector of size (84,).
    """
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:
        results = hands.process(image_rgb)

        features = landmarks_to_features(results)

        if debug:
            annotated_image = image.copy()
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
            cv2.imwrite("./annotated_image.png", annotated_image)

    return features


def preprocess_dataset(
        dataset_path: str | PathLike,
        out_path: str | PathLike
    ) -> None:
    """Preprocess the dataset and save the features to a CSV file.

    Args:
        dataset_path (str | PathLike): The path to the dataset.
        out_path (str | PathLike): The path to save the processed features.
    """
    LABELS = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
        'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
        'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ]

    all_features = []

    for label in LABELS: 
        for i in tqdm(range(1200), desc=f"Label {label}"):
            try:
                features = preprocess_image(f"{dataset_path}/data/{label}/{i}.jpg")
                feature_with_label = features.tolist() + [ord(label) - 49]
                all_features.append(feature_with_label)
            except Exception as e:
                print(f"Skipping {label}/{i}.jpg: {e}")
                continue

    all_features = np.array(all_features, dtype=np.float32)
    columns = [f"f{j}" for j in range(84)] + ["label"]
    features_df = pd.DataFrame(all_features, columns=columns)

    features_df.to_csv(f"{out_path}/hand_landmarks_features.csv", index=False)

    print("Dataset shape:", features_df.shape)

if __name__ == "__main__":
    dataset_path = get_kaggle_dataset("./data/raw/")
    out_path = "./data/processed"
    preprocess_dataset(dataset_path, out_path)
