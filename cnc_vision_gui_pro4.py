import cv2
import numpy as np
import ezdxf
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import time
import subprocess
import threading

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

def build_camera_index_map():
    camera_map = {}
    for index in range(10):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                camera_map[index] = f"Camera {index}"
            cap.release()
    ffmpeg_cams = list_ffmpeg_cameras()
    for idx, name in enumerate(ffmpeg_cams):
        camera_map[idx] = name
    return camera_map

# === GUI Application ===
class CNCVisionApp:
    def __init__(self, master):
        self.master = master
        master.title("CNC Vision Pro Dual Preview with Sliders")
        master.geometry("950x750")

        self.image_path = None
        self.edge_image = None
        self.selected_camera = tk.StringVar()
        self.inches_per_pixel = tk.DoubleVar(value=0.04)
        self.canny_low = tk.IntVar(value=50)
        self.canny_high = tk.IntVar(value=150)
        self.capture_directory = "captures"
        os.makedirs(self.capture_directory, exist_ok=True)

        self.camera_index_map = build_camera_index_map()
        self.available_cameras = list(self.camera_index_map.values())

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

        # Sliders for Canny thresholds
        slider_frame = tk.Frame(master)
        slider_frame.pack(pady=5)
        tk.Label(slider_frame, text="Canny Lower Threshold").pack()
        tk.Scale(slider_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.canny_low, command=lambda _: self.refresh_preview()).pack()
        tk.Label(slider_frame, text="Canny Upper Threshold").pack()
        tk.Scale(slider_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.canny_high, command=lambda _: self.refresh_preview()).pack()

        # Previews
        self.preview_frame = tk.Frame(master)
        self.preview_frame.pack(pady=5)
        self.preview_label_original = tk.Label(self.preview_frame)
        self.preview_label_original.pack(side=tk.LEFT, padx=5)
        self.preview_label_edges = tk.Label(self.preview_frame)
        self.preview_label_edges.pack(side=tk.RIGHT, padx=5)

        # Controls
        tk.Button(master, text="Capture Latest Image", command=self.capture_image).pack(pady=5)
        tk.Button(master, text="Auto-Load Latest Capture", command=self.load_latest_capture).pack(pady=5)
        tk.Button(master, text="Generate Simplified DXF", command=self.process_image).pack(pady=5)

        self.status_label = tk.Label(master, text="", fg="blue")
        self.status_label.pack(pady=10)

        self.cap = None
        self.preview_running = False
        self.preview_thread = None

        self.open_live_preview()

    def open_live_preview(self):
        self.close_preview()

        target_camera_name = self.selected_camera.get()
        target_index = None

        for index, name in self.camera_index_map.items():
            if name == target_camera_name:
                target_index = index
                break

        if target_index is not None:
            cap = cv2.VideoCapture(target_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                self.cap = cap
                self.preview_running = True
                self.preview_thread = threading.Thread(target=self.update_preview)
                self.preview_thread.daemon = True
                self.preview_thread.start()
                self.status_label.config(text=f"Live preview: {target_camera_name}")
            else:
                self.status_label.config(text=f"Failed to open live preview for: {target_camera_name}")
        else:
            self.status_label.config(text=f"No matching index for: {target_camera_name}")

    def close_preview(self):
        self.preview_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def update_preview(self):
        while self.preview_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.process_and_display_frame(frame)
            time.sleep(0.03)  # ~30 FPS

    def process_and_display_frame(self, frame):
        frame_resized = cv2.resize(frame, (400, 300))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Update original preview
        img_original = Image.fromarray(frame_rgb)
        imgtk_original = ImageTk.PhotoImage(image=img_original)
        self.preview_label_original.imgtk = imgtk_original
        self.preview_label_original.configure(image=imgtk_original)

        # Update edge preview with current slider values
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low.get(), self.canny_high.get())
        img_edges = Image.fromarray(edges)
        imgtk_edges = ImageTk.PhotoImage(image=img_edges)
        self.preview_label_edges.imgtk = imgtk_edges
        self.preview_label_edges.configure(image=imgtk_edges)

    def refresh_preview(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.process_and_display_frame(frame)

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
        self.close_preview()
        timestamp = int(time.time())
        image_path = os.path.join(self.capture_directory, f"capture_{timestamp}.jpg")
        edges_path = os.path.join(self.capture_directory, f"capture_{timestamp}_edges.jpg")

        success = self.ffmpeg_capture(image_path)

        if success:
            frame = cv2.imread(image_path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, self.canny_low.get(), self.canny_high.get())

            cv2.imwrite(edges_path, edges)
            self.image_path = edges_path
            self.edge_image = edges_path

            self.status_label.config(text=f"Captured and saved: {edges_path}")

        self.open_live_preview()

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
        self.open_live_preview()

# === Run the app ===
if __name__ == "__main__":
    root = tk.Tk()
    app = CNCVisionApp(root)
    root.mainloop()
