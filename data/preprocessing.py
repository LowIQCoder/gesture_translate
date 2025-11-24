from os import PathLike
import os
import json

from tqdm import tqdm
import numpy as np
import pandas as pd
import ijson
import pyarrow

import cv2
import mediapipe as mp

from sklearn.model_selection import train_test_split
from pathlib import Path

from data.features import landmarks_to_features

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*SymbolDatabase.GetPrototype.*")

def preprocess_image(image_path: PathLike, hands_model=None):
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

def add_noise(gesture, mean, std):
    new_gesture = np.copy(gesture)
    for frame in new_gesture:
        noise = np.random.normal(mean, std, 84)
        frame += noise
    return new_gesture

def random_translate(gesture):
    new_gesture = np.copy(gesture)
    x_translate = np.random.rand()
    y_translate = np.random.rand()
    for frame in new_gesture:
        frame[:42] += x_translate
        frame[42:84] += y_translate
    return new_gesture

def preprocess_slovo_dataset(
        dataset_path: str | PathLike,
        out_path: str | PathLike,
        noise_items=5,
        translate_items=5
    ) -> None:
    """Preprocess the dataset and save the features to CSV files.

    Args:
        dataset_path (str | PathLike): The path to the dataset.
        out_path (str | PathLike): The path to save the processed features.
    """
    # Attachment UUID to categorical label
    labels_df = pd.read_csv("./data/raw/annotations.csv")  
    lab2id = {s: i for i, s in enumerate(labels_df['text'].unique())}
    uuid2lab = {row.attachment_id: lab2id[row.text] for row in labels_df.itertuples(index=False)}

    # Label to category index
    lab2id_df = pd.DataFrame([list(lab2id.keys()), list(lab2id.values())]).transpose()
    lab2id_df.to_csv("./data/processed/labels.csv")

    # For correct data split - guarantee samples per split
    test_samples_per_label = 2
    val_samples_per_label = 2  # Same as test set
    label_counters = {label_id: {'test': 0, 'val': 0, 'train': 0} for label_id in lab2id.values()}

    # Processing landmarks
    train_data = []
    val_data = []  # New validation set
    test_data = []
    augmented_train = []
    augmented_val = []  # Augmented validation data
    augmented_test = []
    
    with open("./data/raw/slovo_mediapipe.json", "r") as input_file:
        for gesture_id, sequence in tqdm(ijson.kvitems(input_file, ""), desc="Processing landmarks", total=20000):
            if len(sequence) <= 0:
                continue
            if uuid2lab[gesture_id] not in list(range(500)):
                continue
            frames = []
            # Extracting each feature
            for frame in sequence:
                frame_landmarks = np.zeros(84, dtype=np.float16)

                if 'hand 1' in frame:
                    frame_landmarks[0:21] = [lm['x'] for lm in frame['hand 1']]
                    frame_landmarks[21:42] = [lm['y'] for lm in frame['hand 1']]

                if 'hand 2' in frame:
                    frame_landmarks[42:63] = [lm['x'] for lm in frame['hand 2']]
                    frame_landmarks[63:84] = [lm['y'] for lm in frame['hand 2']]

                frames.append(frame_landmarks)

            gesture_landmarks = np.vstack(frames) if frames else np.zeros((0, 84), dtype=np.float32)
            label = uuid2lab[gesture_id]

            # Split logic: test -> val -> train
            if label_counters[label]['test'] < test_samples_per_label:
                # Assign to test set
                test_data.append((gesture_landmarks, label))
                label_counters[label]['test'] += 1
                
                # Augment test data
                for _ in range(translate_items):
                    translated = random_translate(gesture_landmarks)
                    augmented_test.append((translated, label))
                for _ in range(noise_items):
                    noise_gesture = add_noise(gesture_landmarks, 0.002, 0.003)
                    augmented_test.append((noise_gesture, label))     
            elif label_counters[label]['val'] < val_samples_per_label:
                # Assign to validation set
                val_data.append((gesture_landmarks, label))
                label_counters[label]['val'] += 1
                
                # Augment validation data
                for _ in range(translate_items):
                    translated = random_translate(gesture_landmarks)
                    augmented_val.append((translated, label))
                for _ in range(noise_items):
                    noise_gesture = add_noise(gesture_landmarks, 0.002, 0.003)
                    augmented_val.append((noise_gesture, label))     
            else:
                # Assign to training set
                train_data.append((gesture_landmarks, label))
                label_counters[label]['train'] += 1
                
                # Augment training data
                for _ in range(translate_items):
                    translated = random_translate(gesture_landmarks)
                    augmented_train.append((translated, label))
                for _ in range(noise_items):
                    noise_gesture = add_noise(gesture_landmarks, 0.002, 0.003)
                    augmented_train.append((noise_gesture, label))

    # Combine original data with augmented data
    train_data = train_data + augmented_train    
    val_data = val_data + augmented_val
    test_data = test_data + augmented_test

    # Convert to DataFrames and save
    # Training set
    train_data_list = [(gesture.tolist(), label) for gesture, label in train_data]
    train_df = pd.DataFrame(train_data_list, columns=["features", "label"])
    train_df.to_parquet("./data/processed/train.parquet", engine="pyarrow")

    # Validation set
    val_data_list = [(gesture.tolist(), label) for gesture, label in val_data]
    val_df = pd.DataFrame(val_data_list, columns=["features", "label"])
    val_df.to_parquet("./data/processed/val.parquet", engine="pyarrow")

    # Test set
    test_data_list = [(gesture.tolist(), label) for gesture, label in test_data]
    test_df = pd.DataFrame(test_data_list, columns=["features", "label"])
    test_df.to_parquet("./data/processed/test.parquet", engine="pyarrow")

    # Print dataset statistics
    print_dataset_statistics(label_counters, lab2id, train_data, val_data, test_data)

def print_dataset_statistics(label_counters, lab2id, train_data, val_data, test_data):
    """Print detailed statistics about the dataset split"""
    
    print("\n=== Dataset Split Statistics ===")
    print(f"Total classes: {len(lab2id)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Total samples: {len(train_data) + len(val_data) + len(test_data)}")
    
    print("\n=== Samples per Class ===")
    for label_id in sorted(lab2id.values()):
        label_name = list(lab2id.keys())[list(lab2id.values()).index(label_id)]
        train_count = label_counters[label_id]['train']
        val_count = label_counters[label_id]['val']
        test_count = label_counters[label_id]['test']
        total_count = train_count + val_count + test_count
        
        print(f"Class {label_id} ({label_name}): "
              f"Train={train_count}, Val={val_count}, Test={test_count}, Total={total_count}")

def preprocess_asl_dataset(
    dataset_path: str | os.PathLike,
    final_path: str | os.PathLike
) -> None:
    """Preprocesses ASL dataset

    Args:
        dataset_path (str | os.PathLike): Path to raw files
        final_path (str | os.PathLike): Path to preprocessed files
    """
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

        # Preprocess images and store results
        for path in tqdm(train_paths, desc=f"Preprocessing Label {label} Train"):
            # TODO Add data augmentations
            features = preprocess_image(path, hands_model)
            train_df.loc[len(train_df)] = ({"features": features, "label": lab2id[label]})

            noised = add_noise(features, 0.1, 0.05)
            moved = random_translate(features)
            train_df.loc[len(train_df)] = ({"features": noised, "label": lab2id[label]})
            train_df.loc[len(train_df)] = ({"features": moved, "label": lab2id[label]})

        for path in tqdm(val_paths, desc=f"Preprocessing Label {label} Val"):
            features = preprocess_image(path, hands_model)
            val_df.loc[len(val_df)] = ({"features": features, "label": lab2id[label]})

        for path in tqdm(test_paths, desc=f"Preprocessing Label {label} Test"):
            features = preprocess_image(path, hands_model)
            test_df.loc[len(test_df)] = ({"features": features, "label": lab2id[label]})

    test_df.to_parquet(os.path.join(final_path, "test.parquet"), index=False)
    train_df.to_parquet(os.path.join(final_path, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(final_path, "val.parquet"), index=False)

if __name__ == "__main__":
    dataset_path = "./data/raw/data"
    out_path = "./data/processed"
    preprocess_asl_dataset(dataset_path, out_path)
