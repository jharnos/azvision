import cv2
import numpy as np
import ezdxf
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import time

# === Utility functions ===

def simplify_contour(contour, tolerance=2.0):
    # Simplify contour using Ramer-Douglas-Peucker algorithm
    epsilon = tolerance * cv2.arcLength(contour, True) / 100.0
    return cv2.approxPolyDP(contour, epsilon, True)

def get_latest_image(directory):
    # Get the latest file from the directory
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.jpg', '.png'))]
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime)
    return latest_file

# === GUI Application ===

class CNCVisionApp:
    def __init__(self, master):
        self.master = master
        master.title("CNC Vision Pro")
        master.geometry("600x500")

        self.image_path = None
        self.inches_per_pixel = tk.DoubleVar(value=0.04)
        self.camera_index = 0
        self.capture_directory = "captures"
        os.makedirs(self.capture_directory, exist_ok=True)

        # Widgets
        tk.Label(master, text="Inches per Pixel:").pack(pady=5)
        self.scale_entry = tk.Entry(master, textvariable=self.inches_per_pixel)
        self.scale_entry.pack(pady=5)

        self.preview_label = tk.Label(master)
        self.preview_label.pack(pady=5)

        self.capture_button = tk.Button(master, text="Capture Latest Image", command=self.capture_image)
        self.capture_button.pack(pady=5)

        self.select_button = tk.Button(master, text="Auto-Load Latest Capture", command=self.load_latest_capture)
        self.select_button.pack(pady=5)

        self.process_button = tk.Button(master, text="Generate Simplified DXF", command=self.process_image)
        self.process_button.pack(pady=5)

        self.status_label = tk.Label(master, text="", fg="blue")
        self.status_label.pack(pady=10)

        # Start live preview
        self.cap = cv2.VideoCapture(self.camera_index)
        self.update_preview()

    def update_preview(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (400, 300))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.preview_label.imgtk = imgtk
            self.preview_label.configure(image=imgtk)
        self.master.after(30, self.update_preview)

    def capture_image(self):
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image from camera.")
            return
        timestamp = int(time.time())
        image_path = os.path.join(self.capture_directory, f"capture_{timestamp}.jpg")
        edges_path = os.path.join(self.capture_directory, f"capture_{timestamp}_edges.jpg")

        # Preprocess and save edge image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        cv2.imwrite(image_path, frame)
        cv2.imwrite(edges_path, edges)
        self.status_label.config(text=f"Image saved: {edges_path}")

        # Auto-load this image for processing
        self.image_path = edges_path

    def load_latest_capture(self):
        latest = get_latest_image(self.capture_directory)
        if latest:
            self.image_path = latest
            self.status_label.config(text=f"Loaded: {latest}")
        else:
            messagebox.showwarning("Warning", "No captures found!")

    def process_image(self):
        if not self.image_path:
            messagebox.showerror("Error", "No image selected or captured.")
            return

        inches_per_pixel = self.inches_per_pixel.get()
        if inches_per_pixel <= 0:
            messagebox.showerror("Error", "Inches per pixel must be greater than 0.")
            return

        try:
            # Process the image
            image = cv2.imread(self.image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                messagebox.showwarning("No Contours", "No shapes found in the image.")
                return

            # Create DXF
            doc = ezdxf.new()
            msp = doc.modelspace()

            for contour in contours:
                if len(contour) < 5:
                    continue

                simplified = simplify_contour(contour, tolerance=2.0)

                points = []
                for point in simplified:
                    x_pixel, y_pixel = point[0]
                    x_real = x_pixel * inches_per_pixel
                    y_real = y_pixel * inches_per_pixel
                    points.append((x_real, y_real))

                msp.add_lwpolyline(points, close=True)

            output_path = filedialog.asksaveasfilename(defaultextension=".dxf", filetypes=[("DXF files", "*.dxf")])
            if output_path:
                doc.saveas(output_path)
                messagebox.showinfo("Success", f"DXF saved to:\n{output_path}")
                self.status_label.config(text="DXF generated successfully!")

        except Exception as e:
            messagebox.showerror("Error", str(e))

# === Run the app ===

if __name__ == "__main__":
    root = tk.Tk()
    app = CNCVisionApp(root)
    root.mainloop()
