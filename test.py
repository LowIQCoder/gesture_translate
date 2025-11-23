import cv2
import mediapipe as mp
import numpy as np
import time
import onnxruntime as ort
from data.features import landmarks_to_features
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

class GestureRecognizer:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.idx2label = pd.read_csv("./data/processed/labels.csv")["0"].tolist()

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.frame_buffer = []
        self.sequence_length = 120
        self.feature_size = 84

    def process_frame(self, frame) -> dict:
        """Process a single frame and add to buffer"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            features = landmarks_to_features(results)

            # Draw landmarks on the frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )

            if features is not None and len(features) == self.feature_size:
                self.frame_buffer.append(features)
                if len(self.frame_buffer) > self.sequence_length:
                    self.frame_buffer.pop(0)
                return {'status': 'frame_added', 'buffer_size': len(self.frame_buffer)}
            else:
                return {'status': 'no_hands_detected', 'buffer_size': len(self.frame_buffer)}

        except Exception as e:
            return {'error': str(e)}

    def predict_gesture(self) -> dict:
        """Run inference when buffer is full"""
        if len(self.frame_buffer) < self.sequence_length:
            return {'error': f'Not enough frames. Need {self.sequence_length}, have {len(self.frame_buffer)}'}

        preprocessing_start = time.time()
        input_tensor = np.expand_dims(np.array(self.frame_buffer, dtype=np.float32), axis=0)
        preprocessing_time = (time.time() - preprocessing_start) * 1000

        inference_start = time.time()
        outputs = self.session.run(None, {self.input_name: input_tensor})
        inference_time = (time.time() - inference_start) * 1000

        logits = outputs[0]
        if logits.ndim == 2 and logits.shape[0] == 1:
            logits = logits[0]

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        pred_idx = int(np.argmax(probs))
        pred_label = self.idx2label[pred_idx]
        confidence = float(probs[pred_idx] * 100)

        # Top-3 predictions
        top_indices = np.argsort(probs)[-3:][::-1]
        top_predictions = [(self.idx2label[idx], float(probs[idx] * 100)) for idx in top_indices]

        self.frame_buffer = []

        return {
            'gesture': pred_label,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'preprocessing_time': preprocessing_time,
            'inference_time': inference_time
        }

def main():
    recognizer = GestureRecognizer("./data/models/gesture_transformer.onnx")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting gesture recognition. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = recognizer.process_frame(frame)

        # Predict if buffer is full
        if len(recognizer.frame_buffer) == recognizer.sequence_length:
            prediction = recognizer.predict_gesture()
            print(f"Predicted Gesture: {prediction['gesture']}, Confidence: {prediction['confidence']:.2f}%")
            print("Top-3 predictions:", prediction['top_predictions'])
            cv2.putText(frame, f"{prediction['gesture']} ({prediction['confidence']:.1f}%)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show webcam feed
        cv2.imshow("Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
