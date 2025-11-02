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
    df = pd.DataFrame(columns=["features", "key"])
    with open(dataset_path + "/slovo_mediapipe.json", "r") as f:
        data = json.load(f)
        
        for key in tqdm(list(data.keys())):
            all_features = list()
            for frame in data[key]:
                features = np.zeros(84, dtype=np.float16)
                try:
                    h1_land = frame['hand 1']
                    f = list()
                    for a in h1_land:
                        f.append(a['x'])
                        f.append(a['y'])
                    features[0: 42] = f
                except:
                    pass
                try:
                    h2_land = frame['hand 2']
                    f = list()
                    for a in h2_land:
                        f.append(a['x'])
                        f.append(a['y'])
                    features[42: 84] = f
                except:
                    pass
                all_features.append(features)
            df.loc[len(df)] = [all_features, key]
            
    lab = pd.read_csv(dataset_path + "/annotations.csv", sep="\t")
    df = pd.merge(df, lab, left_on="key", right_on="attachment_id")
    
    lab2id = {s:i for i, s in enumerate(lab['text'].unique())}
    
    df['label'] = df['text'].apply(lambda x: lab2id[x])
    df = df[['features', 'label']]
    
    lab2id_df = pd.DataFrame([list(lab2id.keys()), list(lab2id.values())]).transpose()
    lab2id_df.to_csv(out_path + '/labels.csv')
    df.to_csv(out_path + '/features.csv')
    print(df.head())

if __name__ == "__main__":
    # dataset_path = get_kaggle_dataset("./data/raw")
    dataset_path = "./data/raw"
    out_path = "./data/processed"
    preprocess_dataset(dataset_path, out_path)
