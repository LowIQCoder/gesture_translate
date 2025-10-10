import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk

class GestureGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gesture Recognition")
        self.root.geometry("800x600")
        
        self.setup_ui()
        
    def setup_ui(self):
        # Основной фрейм
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Левая панель - камера
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.camera_label = ttk.Label(left_frame, text="Camera Feed")
        self.camera_label.pack()
        
        # Правая панель - информация
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Статистика
        stats_frame = ttk.LabelFrame(right_frame, text="Statistics")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.fps_label = ttk.Label(stats_frame, text="FPS: 0")
        self.fps_label.pack(anchor=tk.W)
        
        self.inference_label = ttk.Label(stats_frame, text="Inference: 0ms")
        self.inference_label.pack(anchor=tk.W)
        
        self.preprocessing_label = ttk.Label(stats_frame, text="Preprocessing: 0ms")
        self.preprocessing_label.pack(anchor=tk.W)
        
        # Результаты распознавания
        results_frame = ttk.LabelFrame(right_frame, text="Recognition Results")
        results_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.gesture_label = ttk.Label(results_frame, text="Gesture: None", font=("Arial", 12, "bold"))
        self.gesture_label.pack(anchor=tk.W)
        
        self.confidence_label = ttk.Label(results_frame, text="Confidence: 0%")
        self.confidence_label.pack(anchor=tk.W)
        
        # Embeddings
        embeddings_frame = ttk.LabelFrame(right_frame, text="Embeddings")
        embeddings_frame.pack(fill=tk.BOTH, expand=True)
        
        self.embeddings_text = tk.Text(embeddings_frame, height=10, width=30)
        self.embeddings_text.pack(fill=tk.BOTH, expand=True)
        
    def update_camera(self, frame):
        """Обновить изображение камеры"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tk = ImageTk.PhotoImage(frame_pil)
        
        self.camera_label.config(image=frame_tk)
        self.camera_label.image = frame_tk
        
    def update_stats(self, fps, inference_time, preprocessing_time):
        """Обновить статистику"""
        self.fps_label.config(text=f"FPS: {fps}")
        self.inference_label.config(text=f"Inference: {inference_time:.1f}ms")
        self.preprocessing_label.config(text=f"Preprocessing: {preprocessing_time:.1f}ms")
        
    def update_results(self, gesture, confidence, embeddings):
        """Обновить результаты распознавания"""
        self.gesture_label.config(text=f"Gesture: {gesture}")
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
        
        # Показать топ-3 предсказания
        embeddings_text = ""
        for char, prob in embeddings:
            embeddings_text += f"{char}: {prob:.3f}\n"
        self.embeddings_text.delete(1.0, tk.END)
        self.embeddings_text.insert(1.0, embeddings_text)
        
    def clear_results(self):
        """Очистить результаты"""
        self.gesture_label.config(text="Gesture: None")
        self.confidence_label.config(text="Confidence: 0%")
        self.embeddings_text.delete(1.0, tk.END)
        
    def run(self):
        """Запустить GUI"""
        self.root.mainloop()
        
    def cleanup(self):
        """Очистить ресурсы"""
        self.root.quit()
