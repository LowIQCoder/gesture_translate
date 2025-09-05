import cv2
import mediapipe as mp
import time
import tkinter as tk
from gui import HandTrackingGUI
import numpy as np

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def landmarks_to_features(hand_landmarks):
    """Generate a feature vector from hand landmarks.

    Args:
        hand_landmarks (HandLandmark): The hand landmarks.

    Returns:
        np.ndarray: The feature vector of size (42,).
    """
    coords = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
    
    return coords.flatten()  # shape (42,)


def log_landmarks(hand_landmarks):
    """Logs all landmark coordinates with human-readable names"""
    for idx, landmark in enumerate(hand_landmarks.landmark):
        name = LANDMARK_NAMES[idx]
        print(f"{name} (idx={idx}): x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f}")

def log_features(hand_landmarks):
    """Logs the feature vector derived from hand landmarks"""
    features = landmarks_to_features(hand_landmarks)
    print(f"Feature vector (shape={features.shape}): {features}")

def draw_landmarks_on_frame(frame, hand_landmarks):
    """Draws hand landmarks on the frame"""
    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )

LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]


def run():
    root = tk.Tk()
    gui = HandTrackingGUI(root)

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    prev_time = 0

    def update_frame():
        nonlocal prev_time

        success, frame = cap.read()
        if not success:
            root.after(10, update_frame)
            return

        # Preprocessing timer
        start = time.perf_counter()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Draw landmarks
        text_info = ""
        embeddings = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                embeddings = landmarks_to_features(hand_landmarks)
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    text_info += f"{LANDMARK_NAMES[idx]}: ({landmark.x:.2f}, {landmark.y:.2f}, {landmark.z:.2f})\n"

        # FPS calculation
        end = time.perf_counter()
        render_time_ms = (end - start) * 1000
        fps = 1 / (end - prev_time) if prev_time > 0 else 0
        prev_time = end

        # Update GUI
        frame = cv2.flip(frame, 1)
        gui.update_video(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        gui.update_landmarks_info(text_info, embeddings)
        gui.update_status(fps, render_time_ms)

        root.after(10, update_frame)

    update_frame()
    root.mainloop()


if __name__ == "__main__":
    run()