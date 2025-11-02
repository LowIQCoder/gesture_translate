from os import PathLike
import os

from tqdm import tqdm
import numpy as np
import pandas as pd
import ijson
import pyarrow

import cv2
import mediapipe as mp

from data.features import landmarks_to_features

def preprocess_image(image_path: PathLike, hands_model=None):
    """Preprocess the image for hand tracking and return landmark features.

    Args:
        image_path (str): The path to the image file.
        debug (bool, optional): Whether to display debug info. Defaults to False.

    Raises:
        ValueError: If the image cannot be loaded.

    Returns:
        np.ndarray: The processed feature vector of size (84,).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands_model.process(image_rgb)
    
    # mp_drawing = mp.solutions.drawing_utils
    # mp_drawing_styles = mp.solutions.drawing_styles
    # annotated_image = image.copy()
    # if results.multi_hand_landmarks:
    #     for hand_landmarks in results.multi_hand_landmarks:
    #         mp_drawing.draw_landmarks(
    #             annotated_image,
    #             hand_landmarks,
    #             mp_hands.HAND_CONNECTIONS,
    #             mp_drawing_styles.get_default_hand_landmarks_style(),
    #             mp_drawing_styles.get_default_hand_connections_style()
    #         )
    # cv2.imwrite("./annotated_image.png", annotated_image)

    return landmarks_to_features(results)

def preprocess_video(video_path: PathLike):
    video = cv2.VideoCapture(video_path)
    if video is None:
        raise ValueError(f"Video not found at {video}")
    video.set(cv2.CAP_PROP_FPS, 25)
    features = []
    i = 0
    mp_hands = mp.solutions.hands
    hands_model = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        features.append(preprocess_image(frame, hands_model))
        i += 1
    video.release()
    return features

def preprocess_slovo_dataset(
        dataset_path: str | PathLike,
        out_path: str | PathLike
    ) -> None:
    """Preprocess the dataset and save the features to a CSV file.

    Args:
        dataset_path (str | PathLike): The path to the dataset.
        out_path (str | PathLike): The path to save the processed features.
    """
    # Attechment UUID to categorical label
    labels_df = pd.read_csv("./data/raw/annotations.csv", sep="\t")  
    lab2id = {s:i for i, s in enumerate(labels_df['text'].unique())}
    uuid2lab = {row.attachment_id: lab2id[row.text] for row in labels_df.itertuples(index=False)}

    # Label to category index
    lab2id_df = pd.DataFrame([list(lab2id.keys()), list(lab2id.values())]).transpose()
    lab2id_df.to_csv("./data/processed/labels.csv")

    # Processing landmarks
    data = []
    with open("./data/raw/slovo_mediapipe.json", "r") as input_file:
        for gesture_id, sequence in tqdm(ijson.kvitems(input_file, ""), desc="Processing landmarks", total=20000):
            frames = []
            # Extracting each feature
            for frame in sequence:
                frame_landmarks = np.zeros(84, dtype=np.float32)

                if 'hand 1' in frame:
                    frame_landmarks[0:21] = [lm['x'] for lm in frame['hand 1']]
                    frame_landmarks[21:42] = [lm['y'] for lm in frame['hand 1']]

                if 'hand 2' in frame:
                    frame_landmarks[42:63] = [lm['x'] for lm in frame['hand 2']]
                    frame_landmarks[63:84] = [lm['y'] for lm in frame['hand 2']]

                frames.append(frame_landmarks)

            gesture_landmarks = np.vstack(frames) if frames else np.zeros((0, 84), dtype=np.float32)
            data.append((gesture_landmarks, uuid2lab[gesture_id]))

    
    data_list = [(gesture.tolist(), label) for gesture, label in data]
    result_df = pd.DataFrame(data_list, columns=["features", "label"])
    result_df.to_parquet("./data/processed/features.parquet", engine="pyarrow")

    print(result_df.head())
    print("Total gestures loaded:", len(result_df))


if __name__ == "__main__":
    dataset_path = "./data/raw"
    out_path = "./data/processed"
    preprocess_slovo_dataset(dataset_path, out_path)
    