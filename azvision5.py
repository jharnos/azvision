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
from queue import Queue, Empty

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
    
    # Convert to int to prevent overflow
    h, s, v = int(target_hsv[0]), int(target_hsv[1]), int(target_hsv[2])
    
    # Handle Hue wrapping (since it's circular: 0-180)
    lower_h = max(0, (h - tolerance_h) % 180)
    upper_h = min(180, (h + tolerance_h) % 180)
    
    # Handle Saturation and Value (0-255)
    lower_s = max(0, s - tolerance_s)
    upper_s = min(255, s + tolerance_s)
    lower_v = max(0, v - tolerance_v)
    upper_v = min(255, v + tolerance_v)
    
    # Create the bounds arrays
    lower_bound = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
    upper_bound = np.array([upper_h, upper_s, upper_v], dtype=np.uint8)
    
    # Create the mask
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
    # Apply morphological operations
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Get edges from the mask
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
        self.color_sample_radius = tk.IntVar(value=2)  # Default radius of 2 (5x5 area)

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
        settings_frame = tk.Frame(self.scrollable_frame, bd=2, relief=tk.GROOVE, padx=5, pady=5)
        settings_frame.pack(pady=5, fill=tk.X, expand=True)

        # Create left and right columns
        left_column = tk.Frame(settings_frame)
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        
        right_column = tk.Frame(settings_frame)
        right_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)

        # === Left Column (Canny Edge Settings) ===
        # Basic settings
        basic_settings = tk.LabelFrame(left_column, text="Basic Settings", padx=5, pady=2)
        basic_settings.pack(fill=tk.X, pady=2)
        
        tk.Label(basic_settings, text="Inches per Pixel:").pack(anchor='w')
        tk.Entry(basic_settings, textvariable=self.inches_per_pixel).pack(fill=tk.X)

        # Camera settings
        camera_settings = tk.LabelFrame(left_column, text="Camera Settings", padx=5, pady=2)
        camera_settings.pack(fill=tk.X, pady=2)
        
        tk.Label(camera_settings, text="Select Camera:").pack(anchor='w')
        self.camera_menu = tk.OptionMenu(camera_settings, self.selected_camera, *self.available_cameras, command=self.change_camera)
        self.camera_menu.pack(fill=tk.X)

        tk.Label(camera_settings, text="Select Resolution:").pack(anchor='w')
        resolutions = ["640x480", "1280x720", "1920x1080", "2560x1440"]
        self.resolution_menu = tk.OptionMenu(camera_settings, self.selected_resolution, *resolutions, command=self.change_resolution)
        self.resolution_menu.pack(fill=tk.X)

        self.resolution_label = tk.Label(camera_settings, text="Capture resolution: 1920 x 1080")
        self.resolution_label.pack(anchor='w')

        # Canny Edge settings
        canny_frame = tk.LabelFrame(left_column, text="Canny Edge Detection", padx=5, pady=2)
        canny_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(canny_frame, text="Lower Threshold", height=1).pack(anchor='w')
        tk.Scale(canny_frame, from_=0, to=255, orient=tk.HORIZONTAL, 
                variable=self.canny_low,
                command=lambda _: self.refresh_preview()).pack(fill=tk.X)
        
        tk.Label(canny_frame, text="Upper Threshold", height=1).pack(anchor='w')
        tk.Scale(canny_frame, from_=0, to=255, orient=tk.HORIZONTAL, 
                variable=self.canny_high,
                command=lambda _: self.refresh_preview()).pack(fill=tk.X)

        # === Right Column (Color Detection) ===
        color_frame = tk.LabelFrame(right_column, text="Color Detection", padx=5, pady=2)
        color_frame.pack(fill=tk.X, pady=2)
        
        # Color mode toggle and picker
        mode_frame = tk.Frame(color_frame)
        mode_frame.pack(fill=tk.X)
        
        tk.Checkbutton(mode_frame, text="Use Color Detection", 
                      variable=self.color_mode, 
                      command=self.refresh_preview).pack(side=tk.LEFT)
        
        # Right side: color preview, radius selector, and pick button
        button_frame = tk.Frame(mode_frame)
        button_frame.pack(side=tk.RIGHT)
        
        # Radius selector spinbox
        tk.Label(button_frame, text="Sample R:").pack(side=tk.LEFT)
        radius_spin = tk.Spinbox(button_frame, from_=0, to=10, width=3,
                               textvariable=self.color_sample_radius)
        radius_spin.pack(side=tk.LEFT, padx=2)
        
        # Add color preview canvas
        self.color_preview = tk.Canvas(button_frame, width=20, height=20, 
                                     relief='solid', bd=1)
        self.color_preview.pack(side=tk.LEFT, padx=2)
        # Initialize with gray background
        self.color_preview.configure(bg='#808080')
        
        tk.Button(button_frame, text="Pick Color", 
                 command=self.pick_color).pack(side=tk.LEFT)

        # Color tolerance controls
        tolerance_frame = tk.LabelFrame(color_frame, text="Color Tolerance", padx=5, pady=2)
        tolerance_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(tolerance_frame, text="Hue Tolerance", height=1).pack(anchor='w')
        tk.Scale(tolerance_frame, from_=0, to=90, orient=tk.HORIZONTAL, 
                variable=self.color_tolerance_h,
                command=lambda _: self.refresh_preview()).pack(fill=tk.X)
        
        tk.Label(tolerance_frame, text="Saturation Tolerance", height=1).pack(anchor='w')
        tk.Scale(tolerance_frame, from_=0, to=255, orient=tk.HORIZONTAL, 
                variable=self.color_tolerance_s,
                command=lambda _: self.refresh_preview()).pack(fill=tk.X)
        
        tk.Label(tolerance_frame, text="Value Tolerance", height=1).pack(anchor='w')
        tk.Scale(tolerance_frame, from_=0, to=255, orient=tk.HORIZONTAL, 
                variable=self.color_tolerance_v,
                command=lambda _: self.refresh_preview()).pack(fill=tk.X)

        # === Control Buttons at the bottom ===
        button_frame = tk.Frame(self.scrollable_frame)
        button_frame.pack(pady=2, fill=tk.X)
        tk.Button(button_frame, text="Capture Latest Image", 
                 command=self.capture_image).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        tk.Button(button_frame, text="Auto-Load Latest Capture", 
                 command=self.load_latest_capture).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        tk.Button(button_frame, text="Generate Simplified DXF", 
                 command=self.process_image).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

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
            
            # Create a more visible colored visualization of the mask
            mask_colored = np.zeros_like(frame_resized)
            mask_colored[mask > 0] = [0, 255, 0]  # Make detected areas bright green
            
            # Create the blend using proper numpy arrays
            alpha = 0.7
            mask_blend = cv2.addWeighted(frame_resized, 1.0, mask_colored, 0.5, 0)
            
            # For the additional green tint, create a proper overlay
            green_overlay = np.zeros_like(frame_resized)
            green_overlay[:, :] = [0, 255, 0]  # Fill with green
            
            # Only apply the additional tint to detected areas
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convert mask to 3 channels
            mask_blend = np.where(mask_3ch > 0,
                                cv2.addWeighted(mask_blend, 0.7, green_overlay, 0.3, 0),
                                mask_blend)
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
        except Empty:
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

        try:
            success = self.ffmpeg_capture(image_path)
            if not success:
                raise Exception("Failed to capture image with FFmpeg")

            # Add a timeout for reading the image
            start_time = time.time()
            while not os.path.exists(image_path) and time.time() - start_time < 5:
                time.sleep(0.1)

            if not os.path.exists(image_path):
                raise Exception("Capture file not created within timeout period")

            frame = cv2.imread(image_path)
            if frame is None:
                raise Exception("Failed to read captured image")
            
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

        except Exception as e:
            messagebox.showerror("Capture Error", str(e))
        finally:
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
        
        preview_frame = None  # Store the preview frame globally for the callback
        radius = self.color_sample_radius.get()
        
        def on_mouse_move(event, x, y, flags, param):
            nonlocal preview_frame
            if preview_frame is not None:
                frame_copy = preview_frame.copy()
                cv2.circle(frame_copy, (x, y), radius, (0, 255, 0), 1)
                cv2.imshow("Pick Color", frame_copy)
        
        def on_mouse_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Get the frame and color
                frame = self.frame_buffer[-1]
                orig_height, orig_width = frame.shape[:2]
                preview_height, preview_width = preview_frame.shape[:2]
                
                # Scale the clicked coordinates back to original frame size
                orig_x = int(x * (orig_width / preview_width))
                orig_y = int(y * (orig_height / preview_height))
                
                # Get average color from original frame using scaled radius
                orig_radius = int(radius * (orig_width / preview_width))
                color = self.get_average_color(frame, orig_y, orig_x, orig_radius)
                
                # Use after_idle to update GUI from main thread
                self.master.after_idle(self._update_color_selection, color)
                cv2.destroyWindow("Pick Color")
        
        # Get the frame and resize it to half size
        frame = self.frame_buffer[-1].copy()
        height, width = frame.shape[:2]
        preview_frame = cv2.resize(frame, (width // 2, height // 2))
        
        cv2.namedWindow("Pick Color")
        cv2.setMouseCallback("Pick Color", on_mouse_click)
        cv2.setMouseCallback("Pick Color", lambda *args: (
            on_mouse_move(*args),
            on_mouse_click(*args)
        ))
        cv2.imshow("Pick Color", preview_frame)

    def _update_color_selection(self, color):
        """Update color selection from main thread"""
        self.target_color = color
        self.color_mode.set(True)
        
        # Update color preview
        # Convert BGR to hex color
        hex_color = '#{:02x}{:02x}{:02x}'.format(color[2], color[1], color[0])
        self.color_preview.configure(bg=hex_color)
        
        self.refresh_preview()
        
        # Print color values for debugging
        hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
        print(f"Selected color - BGR: {color}, HSV: {hsv_color}")

    def _on_mousewheel(self, event):
        self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def get_average_color(self, frame, center_y, center_x, radius):
        """Calculate average color in a circular region"""
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        
        # Get colors within the circular mask
        colors = frame[mask == 255]
        
        # Calculate average color if any pixels are selected
        if len(colors) > 0:
            return np.mean(colors, axis=0).astype(np.uint8)
        return frame[center_y, center_x]  # Fallback to single pixel if radius is 0

# === Run the app ===
if __name__ == "__main__":
    root = tk.Tk()
    app = CNCVisionApp(root)
    root.mainloop()