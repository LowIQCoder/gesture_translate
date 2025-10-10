import numpy as np

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