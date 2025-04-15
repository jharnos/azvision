import cv2
import numpy as np
import ezdxf
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import time
import subprocess
import wmi

# === Utility functions ===
def simplify_contour(contour, tolerance=2.0):
    epsilon = tolerance * cv2.arcLength(contour, True) / 100.0
    return cv2.approxPolyDP(contour, epsilon, True)

def get_latest_image(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.jpg', '.png'))]
    if not files:
        return None
    return max(files, key=os.path.getctime)

def list_ffmpeg_cameras():
    devices = []
    try:
        result = subprocess.run(
            ['ffmpeg', '-list_devices', 'true', '-f', 'dshow', '-i', 'dummy'],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
        lines = result.stderr.splitlines()
        for line in lines:
            if '"' in line and '(video)' in line:
                name = line.split('"')[1]
                devices.append(name)
    except Exception as e:
        print(f"Error listing cameras: {e}")
    return devices

# === GUI Application ===
class CNCVisionApp:
    def __init__(self, master):
        self.master = master
        master.title("CNC Vision Pro Final")
        master.geometry("700x620")

        self.image_path = None
        self.selected_camera = tk.StringVar()
        self.inches_per_pixel = tk.DoubleVar(value=0.04)
        self.capture_directory = "captures"
        os.makedirs(self.capture_directory, exist_ok=True)

        self.available_cameras = list_ffmpeg_cameras()
        if not self.available_cameras:
            messagebox.showerror("Camera Error", "No cameras detected via FFmpeg.")
            master.destroy()
            return
        self.selected_camera.set(self.available_cameras[0])

        # GUI layout
        tk.Label(master, text="Inches per Pixel:").pack(pady=5)
        tk.Entry(master, textvariable=self.inches_per_pixel).pack(pady=5)

        tk.Label(master, text="Select Camera:").pack(pady=5)
        self.camera_menu = tk.OptionMenu(master, self.selected_camera, *self.available_cameras, command=self.change_camera)
        self.camera_menu.pack(pady=5)

        self.resolution_label = tk.Label(master, text="Capture resolution: 1920 x 1080")
        self.resolution_label.pack(pady=5)

        self.preview_label = tk.Label(master)
        self.preview_label.pack(pady=5)

        tk.Button(master, text="Capture Latest Image", command=self.capture_image).pack(pady=5)
        tk.Button(master, text="Auto-Load Latest Capture", command=self.load_latest_capture).pack(pady=5)
        tk.Button(master, text="Generate Simplified DXF", command=self.process_image).pack(pady=5)

        self.status_label = tk.Label(master, text="", fg="blue")
        self.status_label.pack(pady=10)

        self.cap = None
        self.open_live_preview()

    def open_live_preview(self):
        # Try opening live preview via OpenCV DirectShow
        try:
            index = 0
            cap_test = None
            for i in range(5):
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and self.selected_camera.get() in self.get_camera_name(cap):
                        cap_test = cap
                        index = i
                        break
                    cap.release()
            if cap_test:
                self.cap = cap_test
                self.status_label.config(text=f"Live preview started (Index {index})")
            else:
                self.cap = None
                self.status_label.config(text="Live preview not available for this camera.")
        except Exception as e:
            print(f"Live preview error: {e}")
            self.cap = None

        self.update_preview()

    def get_camera_name(self, cap):
        # This is a placeholder for getting camera name from cap
        # OpenCV does not directly expose this cleanly
        return self.selected_camera.get()

    def update_preview(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (400, 300))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.preview_label.imgtk = imgtk
                self.preview_label.configure(image=imgtk)
        self.master.after(30, self.update_preview)

    def ffmpeg_capture(self, output_path):
        cmd = [
            'ffmpeg',
            '-f', 'dshow',
            '-video_size', '1920x1080',
            '-i', f'video={self.selected_camera.get()}',
            '-frames:v', '1',
            output_path
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError as e:
            messagebox.showerror("FFmpeg Error", f"Error capturing image with FFmpeg:\n{e.stderr.decode()}")
            return False

    def capture_image(self):
        timestamp = int(time.time())
        image_path = os.path.join(self.capture_directory, f"capture_{timestamp}.jpg")
        edges_path = os.path.join(self.capture_directory, f"capture_{timestamp}_edges.jpg")

        success = self.ffmpeg_capture(image_path)
        if not success:
            return

        frame = cv2.imread(image_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        cv2.imwrite(edges_path, edges)
        self.image_path = edges_path
        self.status_label.config(text=f"Captured and saved: {edges_path}")

    def load_latest_capture(self):
        latest = get_latest_image(self.capture_directory)
        if latest:
            self.image_path = latest
            self.status_label.config(text=f"Loaded image: {latest}")
        else:
            messagebox.showwarning("Warning", "No images found.")

    def process_image(self):
        if not self.image_path:
            messagebox.showerror("Error", "No image loaded or captured.")
            return

        inches_per_pixel = self.inches_per_pixel.get()
        if inches_per_pixel <= 0:
            messagebox.showerror("Error", "Inches per pixel must be greater than 0.")
            return

        try:
            image = cv2.imread(self.image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            doc = ezdxf.new()
            msp = doc.modelspace()

            for contour in contours:
                if len(contour) < 5:
                    continue
                simplified = simplify_contour(contour)
                points = [(pt[0][0] * inches_per_pixel, pt[0][1] * inches_per_pixel) for pt in simplified]
                msp.add_lwpolyline(points, close=True)

            output_path = filedialog.asksaveasfilename(defaultextension=".dxf", filetypes=[("DXF files", "*.dxf")])
            if output_path:
                doc.saveas(output_path)
                messagebox.showinfo("Success", f"DXF saved to: {output_path}")
                self.status_label.config(text="DXF export complete.")

        except Exception as e:
            messagebox.showerror("Processing Error", str(e))

    def change_camera(self, selection):
        self.selected_camera.set(selection)
        if self.cap:
            self.cap.release()
        self.open_live_preview()

# === Run the app ===
if __name__ == "__main__":
    root = tk.Tk()
    app = CNCVisionApp(root)
    root.mainloop()
