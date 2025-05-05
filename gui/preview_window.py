import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import time

class PreviewWindow:
    def __init__(self, parent, cap):
        self.parent = parent
        self.cap = cap
        self.running = True
        
        # Create window
        self.window = tk.Toplevel(parent.master)
        self.window.title("Camera Preview")
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        
        # Create canvas for preview
        self.canvas = tk.Canvas(self.window)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        
        # Start preview thread
        self.preview_thread = threading.Thread(target=self.update_preview)
        self.preview_thread.daemon = True
        self.preview_thread.start()
        
        # Position window
        self.window.geometry("+%d+%d" % (parent.master.winfo_x() + parent.master.winfo_width() + 10,
                                        parent.master.winfo_y()))
        
    def update_preview(self):
        """Update the preview window with camera feed"""
        while self.running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret:
                    # Resize frame to fit window
                    height, width = frame.shape[:2]
                    max_width = 800
                    if width > max_width:
                        scale = max_width / width
                        width = int(width * scale)
                        height = int(height * scale)
                        frame = cv2.resize(frame, (width, height))
                    
                    # Convert to RGB for tkinter
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    photo = ImageTk.PhotoImage(image=image)
                    
                    # Update canvas
                    self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                    self.canvas.photo = photo  # Keep reference
                    
                time.sleep(0.03)  # ~30 FPS
            except Exception as e:
                print(f"Preview error: {e}")
                time.sleep(0.1)
                
    def close(self):
        """Close the preview window"""
        self.running = False
        if self.preview_thread:
            self.preview_thread.join(timeout=1.0)
        self.window.destroy() 