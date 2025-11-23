import numpy as np

def landmarks_to_features(detection_results):
    """
    Generate a feature vector from both hands for a single frame.

    Feature layout (length 84):
        - Right hand x-coordinates: 0-20
        - Right hand y-coordinates: 21-41
        - Left hand x-coordinates: 42-62
        - Left hand y-coordinates: 63-83

    If a hand is missing, its entries remain zeros.

    Args:
        detection_results: MediaPipe Hands results object from hands.process()

    Returns:
        np.ndarray: Feature vector of shape (84,)
    """
    features = np.zeros(84, dtype=np.float32)

    if not detection_results.multi_hand_landmarks:
        return features

    for hand_landmarks, hand_handedness in zip(
        detection_results.multi_hand_landmarks,
        detection_results.multi_handedness
    ):
        # Extract x and y coordinates
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]

        # Assign coordinates to proper section
        if hand_handedness.classification[0].label == "Right":
            features[0:21] = x_coords
            features[21:42] = y_coords
        else:  # Left hand
            features[42:63] = x_coords
            features[63:84] = y_coords

    return features
