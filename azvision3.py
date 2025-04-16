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
from collections import deque
from queue import Queue

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

def color_based_edge_detection(image, target_color, tolerance_h=30, tolerance_s=50, tolerance_v=50):
    """
    image: BGR image from OpenCV
    target_color: tuple of (B,G,R) values
    tolerance_h: Hue tolerance (0-180)
    tolerance_s: Saturation tolerance (0-255)
    tolerance_v: Value tolerance (0-255)
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    target_hsv = cv2.cvtColor(np.uint8([[target_color]]), cv2.COLOR_BGR2HSV)[0][0]
    
    # Handle potential negative values and overflow
    lower_bound = np.array([
        max(0, target_hsv[0] - tolerance_h),
        max(0, target_hsv[1] - tolerance_s),
        max(0, target_hsv[2] - tolerance_v)
    ], dtype=np.uint8)  # Add explicit dtype
    
    upper_bound = np.array([
        min(180, target_hsv[0] + tolerance_h),
        min(255, target_hsv[1] + tolerance_s),
        min(255, target_hsv[2] + tolerance_v)
    ], dtype=np.uint8)  # Add explicit dtype
    
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
    # Fix the morphology operations
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fixed constant name
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Fixed constant name
    
    edges = cv2.Canny(mask, 50, 150)
    
    return edges, mask

class CNCVisionApp:
    def __init__(self, master):
        self.master = master
        master.title("CNC Vision Pro Smooth Preview")
        master.geometry("950x950")

        # Create main canvas with scrollbar
        self.main_canvas = tk.Canvas(master)
        self.scrollbar = tk.Scrollbar(master, orient="vertical", command=self.main_canvas.yview)
        self.scrollable_frame = tk.Frame(self.main_canvas)

        # Configure the canvas
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(
                scrollregion=self.main_canvas.bbox("all")
            )
        )

        # Create a window in the canvas to hold the scrollable frame
        self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)

        # Pack the scrollbar and canvas
        self.scrollbar.pack(side="right", fill="y")
        self.main_canvas.pack(side="left", fill="both", expand=True)

        # Add mousewheel scrolling
        self.main_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Move all variable initializations to the top
        self.image_path = None
        self.edge_image = None
        self.selected_camera = tk.StringVar()
        self.selected_resolution = tk.StringVar(value="1920x1080")
        self.inches_per_pixel = tk.DoubleVar(value=0.04)
        self.canny_low = tk.IntVar(value=50)
        self.canny_high = tk.IntVar(value=150)
        
        # Add the color detection variables here, before creating UI elements
        self.color_mode = tk.BooleanVar(value=False)
        self.target_color = None
        self.color_tolerance_h = tk.IntVar(value=30)
        self.color_tolerance_s = tk.IntVar(value=50)
        self.color_tolerance_v = tk.IntVar(value=50)
        self.last_mask = None

        self.capture_directory = "captures"
        os.makedirs(self.capture_directory, exist_ok=True)

        self.camera_index_map = build_camera_index_map()
        self.available_cameras = list(self.camera_index_map.values())

        if not self.available_cameras:
            messagebox.showerror("Camera Error", "No cameras detected via FFmpeg.")
            master.destroy()
            return

        self.selected_camera.set(self.available_cameras[0])

        # === Layout: Previews at the top ===
        self.preview_frame = tk.Frame(self.scrollable_frame, bd=2, relief=tk.GROOVE)
        self.preview_frame.pack(pady=10)
        self.preview_label_original = tk.Label(self.preview_frame)
        self.preview_label_original.pack(side=tk.LEFT, padx=5, pady=5)
        self.preview_label_edges = tk.Label(self.preview_frame)
        self.preview_label_edges.pack(side=tk.LEFT, padx=5, pady=5)
        self.preview_label_mask = tk.Label(self.preview_frame)
        self.preview_label_mask.pack(side=tk.LEFT, padx=5, pady=5)

        # === Settings below the preview ===
        settings_frame = tk.Frame(self.scrollable_frame, bd=2, relief=tk.GROOVE, padx=10, pady=10)
        settings_frame.pack(pady=10, fill=tk.X, expand=True)

        tk.Label(settings_frame, text="Inches per Pixel:").pack(anchor='w', pady=2)
        tk.Entry(settings_frame, textvariable=self.inches_per_pixel).pack(fill=tk.X, pady=2)

        separator1 = tk.Frame(settings_frame, height=2, bd=1, relief=tk.SUNKEN)
        separator1.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(settings_frame, text="Select Camera:").pack(anchor='w', pady=2)
        self.camera_menu = tk.OptionMenu(settings_frame, self.selected_camera, *self.available_cameras, command=self.change_camera)
        self.camera_menu.pack(fill=tk.X, pady=2)

        tk.Label(settings_frame, text="Select Resolution:").pack(anchor='w', pady=2)
        resolutions = ["640x480", "1280x720", "1920x1080", "2560x1440"]
        self.resolution_menu = tk.OptionMenu(settings_frame, self.selected_resolution, *resolutions, command=self.change_resolution)
        self.resolution_menu.pack(fill=tk.X, pady=2)

        self.resolution_label = tk.Label(settings_frame, text="Capture resolution: 1920 x 1080")
        self.resolution_label.pack(anchor='w', pady=2)

        separator2 = tk.Frame(settings_frame, height=2, bd=1, relief=tk.SUNKEN)
        separator2.pack(fill=tk.X, padx=5, pady=5)

        slider_frame = tk.LabelFrame(settings_frame, text="Canny Edge Detection Thresholds", padx=10, pady=10)
        slider_frame.pack(fill=tk.X, pady=5)
        tk.Label(slider_frame, text="Lower Threshold").pack(anchor='w')
        tk.Scale(slider_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.canny_low, command=lambda _: self.refresh_preview()).pack(fill=tk.X)
        tk.Label(slider_frame, text="Upper Threshold").pack(anchor='w')
        tk.Scale(slider_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.canny_high, command=lambda _: self.refresh_preview()).pack(fill=tk.X)

        # === Color Detection ===
        color_frame = tk.LabelFrame(settings_frame, text="Color Detection", padx=10, pady=10)
        color_frame.pack(fill=tk.X, pady=5)
        
        tk.Checkbutton(color_frame, text="Use Color Detection", 
                       variable=self.color_mode, 
                       command=self.refresh_preview).pack(anchor='w')
        tk.Button(color_frame, text="Pick Color", 
                  command=self.pick_color).pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(color_frame, text="Hue Tolerance").pack(anchor='w')
        tk.Scale(color_frame, from_=0, to=90, orient=tk.HORIZONTAL, 
                 variable=self.color_tolerance_h,
                 command=lambda _: self.refresh_preview()).pack(fill=tk.X)
        
        tk.Label(color_frame, text="Saturation Tolerance").pack(anchor='w')
        tk.Scale(color_frame, from_=0, to=255, orient=tk.HORIZONTAL, 
                 variable=self.color_tolerance_s,
                 command=lambda _: self.refresh_preview()).pack(fill=tk.X)
        
        tk.Label(color_frame, text="Value Tolerance").pack(anchor='w')
        tk.Scale(color_frame, from_=0, to=255, orient=tk.HORIZONTAL, 
                 variable=self.color_tolerance_v,
                 command=lambda _: self.refresh_preview()).pack(fill=tk.X)

        # === Control Buttons ===
        button_frame = tk.Frame(self.scrollable_frame, pady=10)
        button_frame.pack(pady=10, fill=tk.X)
        tk.Button(button_frame, text="Capture Latest Image", command=self.capture_image).pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        tk.Button(button_frame, text="Auto-Load Latest Capture", command=self.load_latest_capture).pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        tk.Button(button_frame, text="Generate Simplified DXF", command=self.process_image).pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        self.status_label = tk.Label(self.scrollable_frame, text="", fg="blue")
        self.status_label.pack(pady=10)

        self.cap = None
        self.preview_running = False
        self.preview_thread = None
        self.frame_buffer = deque(maxlen=2)

        self.update_queue = Queue()
        self.check_queue()  # Start queue checking

        self.open_live_preview()

    # === Application logic ===
    def get_resolution_tuple(self):
        return tuple(map(int, self.selected_resolution.get().split('x')))

    def change_resolution(self, selection):
        self.resolution_label.config(text=f"Capture resolution: {selection.replace('x', ' x ')}")
        self.open_live_preview()

    def change_camera(self, selection):
        self.selected_camera.set(selection)
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
                width, height = self.get_resolution_tuple()
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap = cap
                self.preview_running = True
                self.preview_thread = threading.Thread(target=self.buffered_preview)
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

    def buffered_preview(self):
        while self.preview_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.frame_buffer.append(frame)
            if self.frame_buffer:
                frame_to_process = self.frame_buffer[-1]
                self.process_and_queue_gui_update(frame_to_process)
            time.sleep(0.03)

    def process_and_queue_gui_update(self, frame):
        frame_resized = cv2.resize(frame, (400, 300))

        if self.color_mode.get() and self.target_color is not None:
            edges, mask = color_based_edge_detection(
                frame_resized, 
                self.target_color,
                tolerance_h=self.color_tolerance_h.get(),
                tolerance_s=self.color_tolerance_s.get(),
                tolerance_v=self.color_tolerance_v.get()
            )
            self.last_mask = mask
            
            # Create a colored visualization of the mask
            mask_colored = np.zeros_like(frame_resized)
            mask_colored[mask > 0] = [0, 255, 0]  # Make detected areas green
            
            # Blend with original image for better visualization
            alpha = 0.5
            mask_blend = cv2.addWeighted(frame_resized, alpha, mask_colored, 1-alpha, 0)
        else:
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, self.canny_low.get(), self.canny_high.get())
            self.last_mask = None
            mask_blend = np.zeros_like(frame_resized)

        # Queue the update instead of doing it directly
        self.update_queue.put((frame_resized, edges, mask_blend))

    def check_queue(self):
        """Check for pending GUI updates"""
        try:
            while True:
                # Get all pending updates
                frame, edges, mask = self.update_queue.get_nowait()
                self.update_gui_from_main_thread(frame, edges, mask)
        except Queue.Empty:
            pass
        finally:
            # Schedule the next queue check
            self.master.after(10, self.check_queue)

    def update_gui_from_main_thread(self, frame, edges, mask):
        """Update GUI elements from the main thread"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_original = Image.fromarray(frame_rgb)
        imgtk_original = ImageTk.PhotoImage(image=img_original)
        
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        img_edges = Image.fromarray(edges_rgb)
        imgtk_edges = ImageTk.PhotoImage(image=img_edges)
        
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        img_mask = Image.fromarray(mask_rgb)
        imgtk_mask = ImageTk.PhotoImage(image=img_mask)
        
        self.preview_label_original.imgtk = imgtk_original
        self.preview_label_original.configure(image=imgtk_original)
        self.preview_label_edges.imgtk = imgtk_edges
        self.preview_label_edges.configure(image=imgtk_edges)
        self.preview_label_mask.imgtk = imgtk_mask
        self.preview_label_mask.configure(image=imgtk_mask)

    def refresh_preview(self):
        if self.frame_buffer:
            frame = self.frame_buffer[-1]
            self.process_and_queue_gui_update(frame)

    def ffmpeg_capture(self, output_path):
        width, height = self.get_resolution_tuple()
        resolution_str = f"{width}x{height}"
        cmd = [
            'ffmpeg',
            '-f', 'dshow',
            '-video_size', resolution_str,
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
            
            if self.color_mode.get() and self.target_color is not None:
                edges, _ = color_based_edge_detection(
                    frame,
                    self.target_color,
                    tolerance_h=self.color_tolerance_h.get(),
                    tolerance_s=self.color_tolerance_s.get(),
                    tolerance_v=self.color_tolerance_v.get()
                )
            else:
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
            
            if self.color_mode.get() and self.target_color is not None:
                # Use color-based detection for DXF
                _, mask = color_based_edge_detection(
                    image,
                    self.target_color,
                    tolerance_h=self.color_tolerance_h.get(),
                    tolerance_s=self.color_tolerance_s.get(),
                    tolerance_v=self.color_tolerance_v.get()
                )
                thresh = mask
            else:
                # Use regular edge detection
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

    def pick_color(self):
        if not self.frame_buffer:
            messagebox.showerror("Error", "No image available")
            return
            
        def on_mouse_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                frame = self.frame_buffer[-1]
                self.target_color = frame[y, x]
                cv2.destroyWindow("Pick Color")
                self.refresh_preview()
        
        frame = self.frame_buffer[-1].copy()
        cv2.namedWindow("Pick Color")
        cv2.setMouseCallback("Pick Color", on_mouse_click)
        cv2.imshow("Pick Color", frame)

    def _on_mousewheel(self, event):
        self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

# === Run the app ===
if __name__ == "__main__":
    root = tk.Tk()
    app = CNCVisionApp(root)
    root.mainloop()
