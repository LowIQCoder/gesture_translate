import time
import cv2
import mediapipe as mp
import torch
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from data.features import landmarks_to_features
from src.ml.model import GestureTransformer
import io
from PIL import Image
import json

import os
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))
CORS(app)

class GestureWebApp:
    def __init__(self):
        # Инициализация модели PyTorch
        model_path = "./data/models/best_model.pth"
        config_path = "./data/models/model_configuration.json"
        
        # Загрузка конфигурации модели
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            # Конвертация имен ключей из JSON в параметры модели
            # Примечание: dim_ff в JSON может быть 256, но модель обучена с 512
            model_config = {
                'num_classes': config.get('num_classes', 1001),
                'd_model': config.get('dim_model', 256),
                'd_ff': 512,  # Модель обучена с d_ff=512, несмотря на значение в JSON
                'num_encoders': config.get('num_encoders', 6),
                'nheads': config.get('num_heads', 8),
                'dropout': config.get('dropout', 0.5)
            }
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            # Значения по умолчанию
            model_config = {
                'num_classes': 1001,
                'd_model': 256,
                'd_ff': 256,
                'num_encoders': 6,
                'nheads': 8,
                'dropout': 0.5
            }
        
        # Создание и загрузка модели
        self.model = GestureTransformer(
            num_classes=model_config['num_classes'],
            d_model=model_config['d_model'],
            d_ff=model_config['d_ff'],
            num_encoders=model_config['num_encoders'],
            nheads=model_config['nheads'],
            dropout=model_config['dropout']
        )
        
        # Загрузка весов модели
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            print("Model will be initialized with random weights")
        
        # Инициализация MediaPipe
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Переменные для статистики
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
    def process_frame(self, frame_data):
        """Обработать кадр и вернуть результаты распознавания"""
        try:
            # Декодирование base64 изображения
            image_data = base64.b64decode(frame_data.split(',')[1])
            image = Image.open(io.BytesIO(image_data))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Обработка жестов
            preprocessing_start = time.time()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            preprocessing_time = (time.time() - preprocessing_start) * 1000
            
            # Предсказание
            inference_start = time.time()
            gesture_result = {
                'gesture': None,
                'confidence': 0,
                'embeddings': [],
                'preprocessing_time': preprocessing_time,
                'inference_time': 0
            }
            
            if results.multi_hand_landmarks:
                features = landmarks_to_features(results).reshape((1, 1, 84))
                features_tensor = torch.from_numpy(features).float()
                
                with torch.no_grad():
                    labels = self.model(features_tensor)
                
                probs = torch.softmax(labels, dim=1).squeeze()
                
                pred_label = torch.argmax(probs).item()
                confidence = probs[pred_label].item() * 100
                
                # Топ-3 предсказания
                top_indices = torch.topk(probs, 3).indices.tolist()
                embeddings = [(chr(idx + 49), probs[idx].item()) for idx in top_indices]
                
                # Первое предсказание (самое вероятное)
                top_gesture = chr(top_indices[0] + 49)
                
                gesture_result.update({
                    'gesture': top_gesture,
                    'confidence': confidence,
                    'embeddings': embeddings
                })
            
            inference_time = (time.time() - inference_start) * 1000
            gesture_result['inference_time'] = inference_time
            
            # Обновление FPS
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.start_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.start_time = current_time
            
            gesture_result['fps'] = self.fps
            
            return gesture_result
            
        except Exception as e:
            return {
                'error': str(e),
                'gesture': None,
                'confidence': 0,
                'embeddings': [],
                'preprocessing_time': 0,
                'inference_time': 0,
                'fps': 0
            }

# Глобальный экземпляр приложения
gesture_app = GestureWebApp()

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """API endpoint для обработки кадра"""
    try:
        data = request.get_json()
        frame_data = data.get('frame')
        
        if not frame_data:
            return jsonify({'error': 'No frame data provided'}), 400
        
        result = gesture_app.process_frame(frame_data)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    """Проверка состояния API"""
    return jsonify({'status': 'ok', 'message': 'Gesture recognition API is running'})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=False, host='0.0.0.0', port=port)