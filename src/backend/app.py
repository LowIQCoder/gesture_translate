from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64
import time
import onnxruntime as ort
from data.features import landmarks_to_features
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

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

    def process_frame(self, frame_data: str) -> dict:
        """Process single frame and return prediction"""
        try:
            # Decode base64 image
            image_data = base64.b64decode(frame_data.split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {'error': 'Failed to decode image'}
            
            preprocessing_start = time.time()

            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Extract features
            features = landmarks_to_features(results)
            
            if features is None:
                return {'status': 'no_hands_detected'}
            
            # Ensure features are the right shape for model (batch_size, features)
            input_tensor = np.expand_dims(np.array(features, dtype=np.float32), axis=0)

            preprocessing_time = (time.time() - preprocessing_start) * 1000

            # Run inference
            inference_start = time.time()
            outputs = self.session.run(None, {self.input_name: input_tensor})
            inference_time = (time.time() - inference_start) * 1000

            # Process outputs
            logits = outputs[0]
            if logits.ndim == 2 and logits.shape[0] == 1:
                logits = logits[0]  # remove batch dim

            # Softmax
            exp_logits = np.exp(logits - np.max(logits))  # for numerical stability
            probs = exp_logits / np.sum(exp_logits)

            # Prediction
            pred_idx = int(np.argmax(probs))
            pred_label = self.idx2label[pred_idx]
            confidence = float(probs[pred_idx] * 100)

            # Top-3 predictions
            top_indices = np.argsort(probs)[-3:][::-1]
            top_predictions = [(self.idx2label[idx], float(probs[idx] * 100)) for idx in top_indices]

            return {
                'gesture': pred_label,
                'confidence': confidence,
                'embeddings': top_predictions,
                'preprocessing_time': preprocessing_time,
                'inference_time': inference_time
            }
                
        except Exception as e:
            return {'error': str(e)}


# Initialize recognizer
recognizer = GestureRecognizer("./data/models/gesture_transformer.onnx")

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """Process single frame and return prediction"""
    data = request.get_json()
    frame_data = data.get('frame')
    
    if not frame_data:
        return jsonify({'error': 'No frame data provided'}), 400
    
    # Process frame and get prediction
    result = recognizer.process_frame(frame_data)
    
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
