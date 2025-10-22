import kagglehub
from os import PathLike
import os
import json

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd

import cv2
import mediapipe as mp

from data.features import landmarks_to_features

def get_kaggle_dataset(
        dataset: str
    ) -> str:
    """Downloads dataset from KaggleHub

    Note:
        For correct work of this function make shure that **KAGGLEHUB_CACHE** envoirement is set to "./data/raw/"
    
    Args:
        path (str): Dataset handle on kaggle

    Returns:
        str: Path datased downloaded to
    """
    dataset_path = kagglehub.dataset_download(dataset)
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

def load_frame(image_path_or_frame):
    """Load and convert an image/frame to RGB for processing.

    Args:
        image_path_or_frame (str | np.ndarray): Path to image or a frame array.

    Returns:
        np.ndarray: RGB image.
    """
    if isinstance(image_path_or_frame, str):
        image = cv2.imread(image_path_or_frame)
        if image is None:
            raise ValueError(f"Image not found at {image_path_or_frame}")
    else:
        image = image_path_or_frame

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb, image  # Return original for annotation if needed


def preprocess_frame(frame_rgb, debug=False, orig_image=None, frame_idx=None):
    """Process a frame for hand landmarks and optionally save debug image.

    Args:
        frame_rgb (np.ndarray): RGB frame.
        debug (bool, optional): Whether to save annotated debug image.
        orig_image (np.ndarray, optional): Original BGR frame for debugging.
        frame_idx (int, optional): Frame index for saving debug image.

    Returns:
        np.ndarray: Feature vector (84,).
    """
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:
        results = hands.process(frame_rgb)
        features = landmarks_to_features(results)

        if debug and orig_image is not None:
            annotated_image = orig_image.copy()
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
            cv2.imwrite(f"./data/imgs/processed_{frame_idx:03}.png", annotated_image)

    return features


def preprocess_video(video_path, debug=False):
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_FPS, 25)
    features = []
    i = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_rgb, orig = load_frame(frame)
        features.append(preprocess_frame(frame_rgb, debug=debug, orig_image=orig, frame_idx=i))
        i += 1
    video.release()
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
    with open(f"{dataset_path}/WLASL_v0.3.json", "r") as f:
        words = json.load(f)

    df = pd.DataFrame(columns=["features", "label"])
    for word in tqdm(words):
        for inst in word['instances']:
            try:
                features = preprocess_video(f"{dataset_path}/videos/{inst['video_id']}.mp4")
            except:
                continue
            new_row = pd.DataFrame({"features": features, "label": word['gloss']})
            df = pd.concat([df, new_row])
    df.to_csv(f"{out_path}/video_landmarks.csv")

if __name__ == "__main__":
    # dataset_path = get_kaggle_dataset("risangbaskoro/wlasl-processed")
    # out_path = "./data/processed"
    # preprocess_dataset(dataset_path, out_path)
    dataset_path = get_kaggle_dataset("risangbaskoro/wlasl-processed")
    features = preprocess_video(dataset_path + "/videos/69241.mp4", True)
    print(features)
