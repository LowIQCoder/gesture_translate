import cv2
import mediapipe as mp
import time
from logger import setup_logger
from model import load_model, predict_from_results
from gui import GestureGUI

logger = setup_logger("MainApp")

def main():
    """Main function"""
    # Инициализация
    model, device = load_model("../ml/best_model.pth")
    cam = cv2.VideoCapture(0)
    gui = GestureGUI()
    
    # Инициализация MediaPipe
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Переменные для статистики
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            frame_start = time.time()
            
            # Захват кадра
            ret, frame = cam.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Обработка жестов
            preprocessing_start = time.time()
            results = hands.process(rgb_frame)
            preprocessing_time = (time.time() - preprocessing_start) * 1000
            
            # Предсказание
            inference_start = time.time()
            if results.multi_hand_landmarks:
                pred_label, probs = predict_from_results(model, device, results)
                confidence = max(probs) * 100
                
                # Топ-3 предсказания
                top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
                embeddings = [(chr(idx + 49), probs[idx]) for idx in top_indices]
                
                gui.update_results(pred_label, confidence, embeddings)
            else:
                gui.clear_results()
            
            inference_time = (time.time() - inference_start) * 1000
            
            # Отрисовка landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            
            # Обновление GUI
            gui.update_camera(frame)
            
            # Обновление статистики
            frame_count += 1
            current_time = time.time()
            if current_time - start_time >= 1.0:
                fps = frame_count
                frame_count = 0
                start_time = current_time
                
            gui.update_stats(fps, inference_time, preprocessing_time)
            
            # Обработка событий GUI
            gui.root.update()
            
            # Проверка на закрытие
            if not gui.root.winfo_exists():
                break
                
    finally:
        cam.release()
        cv2.destroyAllWindows()
        gui.cleanup()


if __name__ == "__main__":
    main()
