import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class HandTrackingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Tracking")

        # Main container (2 columns)
        main_frame = tk.Frame(root)
        main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Left side: Video + Options
        left_frame = tk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="n")

        self.video_label = tk.Label(left_frame)
        self.video_label.pack(padx=10, pady=10)

        options_frame = ttk.LabelFrame(left_frame, text="Options")
        options_frame.pack(padx=10, pady=10, fill="x")

        self.option_vars = []
        for i in range(4):
            var = tk.BooleanVar()
            chk = ttk.Checkbutton(options_frame, text=f"Option {i+1}", variable=var)
            chk.pack(anchor="w")
            self.option_vars.append(var)

        # Right side: Landmarks info + Status
        right_frame = tk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="n")

        self.info_text = tk.Text(right_frame, height=21, width=50)
        self.info_text.pack(padx=10, pady=10)

        self.embedding_text = tk.Text(right_frame, height=15, width=50)
        self.embedding_text.pack(padx=10, pady=10)

        self.status_label = tk.Label(right_frame, text="FPS: 0 | Render: 0 ms", font=("Arial", 12))
        self.status_label.pack(padx=10, pady=10)

    def update_video(self, frame_bgr):
        """Update video frame in GUI"""
        img = Image.fromarray(frame_bgr)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def update_landmarks_info(self, text, embeddings):
        """Update landmarks text box"""
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, text if text.strip() else "No landmarks detected")
        self.embedding_text.delete(1.0, tk.END)
        self.embedding_text.insert(tk.END, embeddings if text.strip() else "No landmarks detected")

    def update_status(self, fps, render_time_ms):
        """Update FPS and render time"""
        self.status_label.config(text=f"FPS: {fps:.1f} | Render: {render_time_ms:.2f} ms")
