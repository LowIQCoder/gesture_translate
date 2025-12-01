import os
import sys

from tqdm import tqdm
import numpy as np
import pandas as pd
import pyarrow

import cv2
import mediapipe as mp

from sklearn.model_selection import train_test_split
from pathlib import Path

from data.features import landmarks_to_features

import kagglehub

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*SymbolDatabase.GetPrototype.*")

def preprocess_image(image_path: os.PathLike, hands_model=None):
    """Preprocess the image for hand tracking and return landmark features.

    Args:
        image_path (str): The path to the image file.
        frame (int), optional: If debig is set loggs files with frame number
        debug (bool, optional): Whether to display debug info. Defaults to False.

    Raises:
        ValueError: If the image cannot be loaded.

    Returns:
        np.ndarray: The processed feature vector of size (84,).
    """
    image_path = Path.absolute(Path(image_path))
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Image not found at {image_path}\nPlease make shure that path contains only LATIN letters")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands_model.process(image_rgb)

    return landmarks_to_features(results)

def horizontal_flip(gesture: np.ndarray) -> np.ndarray:
    """Horizontally flip gesture and swap hand positions.

    After flipping, what was right hand becomes left hand and vice versa.

    Args:
        gesture (np.ndarray): Array of features

    Returns:
        np.ndarray: Horizontally flipped features with hands swapped
    """
    flipped_gesture = np.zeros_like(gesture)
    
    # Copy and flip right hand to left hand position
    if not np.allclose(gesture[0:21], 0):  # If right hand exists
        # Flip x-coordinates and place in left hand section
        flipped_gesture[42:63] = 1 - gesture[0:21]    # x-coords
        flipped_gesture[63:84] = gesture[21:42]       # y-coords (unchanged)
    
    # Copy and flip left hand to right hand position
    if not np.allclose(gesture[42:63], 0):  # If left hand exists
        # Flip x-coordinates and place in right hand section
        flipped_gesture[0:21] = 1 - gesture[42:63]    # x-coords
        flipped_gesture[21:42] = gesture[63:84]       # y-coords (unchanged)
    
    return flipped_gesture

def add_noise(
        gesture:np.ndarray, 
        mean: float,
        std: float
    ) -> np.ndarray:
    """Adds Gaussian noise to gesture landmarks

    Args:
        gesture (np.ndarray): Array of features
        mean (float): Mean
        std (float): Std

    Returns:
        np.ndarray: Features with noise applied
    """
    new_gesture = np.copy(gesture)
    noise = np.random.normal(mean, std, 84)
    for i in range(len(gesture)):
        # Checking if hand exists
        if new_gesture[i] != 0:
            new_gesture[i] += noise[i]
    return new_gesture

def random_translate(
        gesture: np.ndarray
    ) -> np.ndarray:
    """Randomly moves gesture landmarks

    Args:
        gesture (np.ndarray): Array of features

    Returns:
        np.ndarray: Features with some translation
    """
    new_gesture = np.copy(gesture)
    x_translate = np.random.rand()
    y_translate = np.random.rand()
    
    # Checking each hand
    if not np.allclose(gesture[:21], 0):
        new_gesture[:21] += x_translate
        new_gesture[21:42] += y_translate
    if not np.allclose(gesture[42:63], 0):
        new_gesture[42:63] += x_translate
        new_gesture[63:84] += y_translate
    
    return new_gesture

def random_scale(
        gesture: np.ndarray
    ) -> np.ndarray:
    """Randomly scales gesture landmarks

    Args:
        gesture (np.ndarray): Array of features

    Returns:
        np.ndarray: Features with some scale
    """
    scale_factor = np.random.rand()
    scaled_gesture = gesture * scale_factor
    
    return scaled_gesture

def random_rotate(
        gesture: np.ndarray
    ) -> np.ndarray:
    """Randomly rotaset gesture. Maximum 15 degree

    Args:
        gesture (np.ndarray): Array of features

    Returns:
        np.ndarray: Features with some rotation
    """
    # Some vars
    angle_rad = np.deg2rad(np.random.uniform(-15, 15))
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    rotation_matrix = np.array([[cos_angle, -sin_angle],
                                 [sin_angle, cos_angle]])

    new_gesture = np.copy(gesture)

    # Rotating gesture
    for hand_start in [0, 42]:
        if not np.allclose(gesture[hand_start:hand_start + 21], 0):
            for i in range(21):
                point = np.array([gesture[hand_start + i], gesture[hand_start + 21 + i]])
                rotated_point = rotation_matrix @ point
                new_gesture[hand_start + i] = rotated_point[0]
                new_gesture[hand_start + 21 + i] = rotated_point[1]

    return new_gesture

def preprocess_asl_dataset(
    dataset_path: str | os.PathLike,
    final_path: str | os.PathLike
) -> None:
    """Preprocesses ASL dataset

    Args:
        dataset_path (str | os.PathLike): Path to raw files
        final_path (str | os.PathLike): Path to preprocessed files
    """
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    
    # Splitting ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Extracting labels (folder names)
    labels = [p.name for p in Path(dataset_path).iterdir() if p.is_dir()]

    # Saving labels to id mapping to labels.csv
    lab2id = pd.DataFrame(
        [{"label": label, "id": idx} for idx, label in enumerate(labels)]
    )
    lab2id.to_csv(os.path.join(final_path, "labels.csv"), index=False)
    lab2id = dict(zip(lab2id["label"], lab2id["id"]))

    # Creating mediapipe model for preprocessing
    mp_hands = mp.solutions.hands
    hands_model = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
    
    # Used augmentations
    augmentations = [
        lambda x: x,
        lambda x: add_noise(x, 0.1, 0.02),
        lambda x: random_translate(x),
        lambda x: random_scale(x),
        lambda x: random_rotate(x)
    ]

    # Preprocessiing images
    train_df = pd.DataFrame(columns=["features", "label"])
    val_df = pd.DataFrame(columns=["features", "label"])
    test_df = pd.DataFrame(columns=["features", "label"])
    for label in labels:
        img_paths = [
            os.path.join(dataset_path, f"{label}/{i}.jpg")
            for i in range(1200)
        ]

        # First split: train vs temp (val + test)
        train_paths, temp_paths = train_test_split(
            img_paths,
            test_size=VAL_RATIO + TEST_RATIO,
            shuffle=True,
            random_state=42
        )

        # Second split: val vs test
        val_paths, test_paths = train_test_split(
            temp_paths,
            test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
            shuffle=True,
            random_state=42
        )

        for path in tqdm(train_paths, desc=f"Preprocessing Label {label} Train"):
            features = preprocess_image(path, hands_model)

            for aug_func in augmentations:
                augmented = aug_func(features)
                augmented_flipped = horizontal_flip(augmented)
                
                train_df.loc[len(train_df)] = ({"features": augmented, "label": lab2id[label]})
                train_df.loc[len(train_df)] = ({"features": augmented_flipped, "label": lab2id[label]})

        for path in tqdm(val_paths, desc=f"Preprocessing Label {label} Val"):
            features = preprocess_image(path, hands_model)
            
            val_df.loc[len(val_df)] = ({"features": features, "label": lab2id[label]})
            val_df.loc[len(val_df)] = ({"features": horizontal_flip(features), "label": lab2id[label]})

        for path in tqdm(test_paths, desc=f"Preprocessing Label {label} Test"):
            features = preprocess_image(path, hands_model)
            
            test_df.loc[len(test_df)] = ({"features": features, "label": lab2id[label]})

    test_df.to_parquet(os.path.join(final_path, "test.parquet"), index=False)
    train_df.to_parquet(os.path.join(final_path, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(final_path, "val.parquet"), index=False)

if __name__ == "__main__":
    # Download latest version
    try:
        dataset_path = sys.argv[1]
    except:
        dataset_path = kagglehub.dataset_download("harshvardhan21/sign-language-detection-using-images")
        dataset_path = os.path.join(dataset_path, "data")
    print(f"Loaded dataset from {dataset_path}")
    out_path = "./data/processed"
    preprocess_asl_dataset(dataset_path, out_path)
