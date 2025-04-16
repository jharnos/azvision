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
def simplify_contour(contour, tolerance=0.1):
    """
    Simplify contour while preserving maximum detail
    tolerance: much lower value for very high detail
    """
    epsilon = tolerance * cv2.arcLength(contour, True) / 200.0  # Doubled precision (was 100.0)
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
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    target_hsv = cv2.cvtColor(np.uint8([[target_color]]), cv2.COLOR_BGR2HSV)[0][0]
    
    h, s, v = int(target_hsv[0]), int(target_hsv[1]), int(target_hsv[2])
    
    lower_h = max(0, (h - tolerance_h) % 180)
    upper_h = min(180, (h + tolerance_h) % 180)
    
    lower_s = max(0, s - tolerance_s)
    upper_s = min(255, s + tolerance_s)
    lower_v = max(0, v - tolerance_v)
    upper_v = min(255, v + tolerance_v)
    
    lower_bound = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
    upper_bound = np.array([upper_h, upper_s, upper_v], dtype=np.uint8)
    
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
    # More aggressive morphological operations
    kernel = np.ones((5,5), np.uint8)  # Larger kernel
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)  # Add dilation
    
    # Ensure mask is binary
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Get edges from the mask
    edges = cv2.Canny(mask, 50, 150)
    
    # Thicken the edges
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    
    return edges, mask

class CNCVisionApp:
    def __init__(self, master):
        self.master = master
        master.title("AriZona Vision")
        
        # Define color scheme
        self.colors = {
            'main': "#81d2c8",      # Main background
            'secondary': "#abe3d6",  # Secondary elements
            'accent1': "#e3c46e",    # Warm gold
            'accent2': "#e34f78",    # Rose
            'text': "#333333"        # Dark gray for text
        }
        
        # Set the background color for the main window
        master.configure(bg=self.colors['main'])
        
        # Create a main frame to hold everything
        main_frame = tk.Frame(master, bg=self.colors['main'])
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Create canvas and scrollbar with the color scheme
        self.main_canvas = tk.Canvas(main_frame, bg=self.colors['main'])
        self.scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=self.main_canvas.yview)
        
        # Create the scrollable frame
        self.scrollable_frame = tk.Frame(self.main_canvas, bg=self.colors['main'])
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )
        
        # Create the window in the canvas
        self.canvas_frame = self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Configure canvas scrolling
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Bind canvas resizing
        self.main_canvas.bind('<Configure>', self.on_canvas_configure)
        
        # Layout canvas and scrollbar
        self.main_canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Configure scrollable frame
        self.scrollable_frame.grid_columnconfigure(0, weight=1)

        # Initialize all variables first
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
        self.preview_frame = tk.Frame(self.scrollable_frame, bd=2, relief=tk.GROOVE, bg=self.colors['secondary'])
        self.preview_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.preview_frame.grid_columnconfigure((0,1,2), weight=1)

        self.preview_label_original = tk.Label(self.preview_frame)
        self.preview_label_original.grid(row=0, column=0, padx=2, pady=2, sticky="nsew")
        
        self.preview_label_edges = tk.Label(self.preview_frame)
        self.preview_label_edges.grid(row=0, column=1, padx=2, pady=2, sticky="nsew")
        
        self.preview_label_mask = tk.Label(self.preview_frame)
        self.preview_label_mask.grid(row=0, column=2, padx=2, pady=2, sticky="nsew")

        # === Settings below the preview ===
        settings_frame = tk.Frame(self.scrollable_frame, bd=2, relief=tk.GROOVE, bg=self.colors['secondary'])
        settings_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        settings_frame.grid_columnconfigure((0,1), weight=1)

        # Create left and right columns
        left_column = tk.Frame(settings_frame, bg=self.colors['main'])
        left_column.grid(row=0, column=0, sticky="nsew", padx=2)
        left_column.grid_columnconfigure(0, weight=1)
        
        right_column = tk.Frame(settings_frame, bg=self.colors['main'])
        right_column.grid(row=0, column=1, sticky="nsew", padx=2)
        right_column.grid_columnconfigure(0, weight=1)

        # === Left Column (Canny Edge Settings) ===
        # Basic settings
        basic_settings = tk.LabelFrame(left_column, text="Basic Settings", **{'bg': self.colors['secondary'], 'fg': self.colors['text'], 'font': ('Arial', 10, 'bold')})
        basic_settings.grid(row=0, column=0, sticky="ew", pady=2)
        basic_settings.grid_columnconfigure(0, weight=1)
        
        tk.Label(basic_settings, text="Inches per Pixel:").grid(row=0, column=0, sticky="w")
        tk.Entry(basic_settings, textvariable=self.inches_per_pixel).grid(row=1, column=0, sticky="ew")

        # Camera settings
        camera_settings = tk.LabelFrame(left_column, text="Camera Settings", **{'bg': self.colors['secondary'], 'fg': self.colors['text'], 'font': ('Arial', 10, 'bold')})
        camera_settings.grid(row=1, column=0, sticky="ew", pady=2)
        camera_settings.grid_columnconfigure(0, weight=1)
        
        tk.Label(camera_settings, text="Select Camera:").grid(row=0, column=0, sticky="w")
        self.camera_menu = tk.OptionMenu(camera_settings, self.selected_camera, *self.available_cameras, 
                                       command=self.change_camera)
        self.camera_menu.grid(row=1, column=0, sticky="ew")

        tk.Label(camera_settings, text="Select Resolution:").grid(row=2, column=0, sticky="w")
        resolutions = ["640x480", "1280x720", "1920x1080", "2560x1440"]
        self.resolution_menu = tk.OptionMenu(camera_settings, self.selected_resolution, *resolutions,
                                           command=self.change_resolution)
        self.resolution_menu.grid(row=3, column=0, sticky="ew")

        self.resolution_label = tk.Label(camera_settings, text="Capture resolution: 1920 x 1080")
        self.resolution_label.grid(row=4, column=0, sticky="w")

        # Check resolution button
        check_res_button = tk.Button(camera_settings, text="Check Available Resolutions",
                                   command=lambda: self.check_current_camera_resolutions())
        check_res_button.grid(row=5, column=0, sticky="ew", pady=2)

        # Canny Edge settings
        canny_frame = tk.LabelFrame(left_column, text="Canny Edge Detection", **{'bg': self.colors['secondary'], 'fg': self.colors['text'], 'font': ('Arial', 10, 'bold')})
        canny_frame.grid(row=2, column=0, sticky="ew", pady=2)
        canny_frame.grid_columnconfigure(0, weight=1)
        
        tk.Label(canny_frame, text="Lower Threshold", height=1).grid(row=0, column=0, sticky="w")
        tk.Scale(canny_frame, from_=0, to=255, orient=tk.HORIZONTAL, 
                 variable=self.canny_low,
                 command=lambda _: self.refresh_preview()).grid(row=1, column=0, sticky="ew")
        
        tk.Label(canny_frame, text="Upper Threshold", height=1).grid(row=2, column=0, sticky="w")
        tk.Scale(canny_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                 variable=self.canny_high,
                 command=lambda _: self.refresh_preview()).grid(row=3, column=0, sticky="ew")

        # === Right Column (Color Detection) ===
        color_frame = tk.LabelFrame(right_column, text="Color Detection", **{'bg': self.colors['secondary'], 'fg': self.colors['text'], 'font': ('Arial', 10, 'bold')})
        color_frame.grid(row=0, column=0, sticky="ew", pady=2)
        color_frame.grid_columnconfigure(0, weight=1)
        
        # Color mode toggle and picker
        mode_frame = tk.Frame(color_frame)
        mode_frame.grid(row=0, column=0, sticky="ew")
        mode_frame.grid_columnconfigure(1, weight=1)
        
        tk.Checkbutton(mode_frame, text="Use Color Detection",
                       variable=self.color_mode,
                       command=self.refresh_preview).grid(row=0, column=0, sticky="w")
        
        picker_frame = tk.Frame(mode_frame)
        picker_frame.grid(row=0, column=1, sticky="e")
        
        tk.Label(picker_frame, text="Sample R:").grid(row=0, column=0)
        tk.Spinbox(picker_frame, from_=0, to=10, width=3,
                   textvariable=self.color_sample_radius).grid(row=0, column=1, padx=2)
        
        self.color_preview = tk.Canvas(picker_frame, width=20, height=20,
                                     relief='solid', bd=1, bg='#808080')
        self.color_preview.grid(row=0, column=2, padx=2)
        
        tk.Button(picker_frame, text="Pick Color",
                  command=self.pick_color).grid(row=0, column=3)

        # Color tolerance controls
        tolerance_frame = tk.LabelFrame(color_frame, text="Color Tolerance", **{'bg': self.colors['secondary'], 'fg': self.colors['text'], 'font': ('Arial', 10, 'bold')})
        tolerance_frame.grid(row=1, column=0, sticky="ew", pady=2)
        tolerance_frame.grid_columnconfigure(0, weight=1)
        
        tk.Label(tolerance_frame, text="Hue Tolerance", height=1).grid(row=0, column=0, sticky="w")
        tk.Scale(tolerance_frame, from_=0, to=90, orient=tk.HORIZONTAL,
                 variable=self.color_tolerance_h,
                 command=lambda _: self.refresh_preview()).grid(row=1, column=0, sticky="ew")
        
        tk.Label(tolerance_frame, text="Saturation Tolerance", height=1).grid(row=2, column=0, sticky="w")
        tk.Scale(tolerance_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                 variable=self.color_tolerance_s,
                 command=lambda _: self.refresh_preview()).grid(row=3, column=0, sticky="ew")
        
        tk.Label(tolerance_frame, text="Value Tolerance", height=1).grid(row=4, column=0, sticky="w")
        tk.Scale(tolerance_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                 variable=self.color_tolerance_v,
                 command=lambda _: self.refresh_preview()).grid(row=5, column=0, sticky="ew")

        # === Control Buttons at the bottom ===
        button_frame = tk.Frame(self.scrollable_frame, bg=self.colors['main'])
        button_frame.grid(row=2, column=0, sticky="ew", pady=2)
        button_frame.grid_columnconfigure((0,1,2), weight=1)
        
        tk.Button(button_frame, text="Capture Latest Image",
                  command=self.capture_image, **{'bg': self.colors['accent2'], 'fg': 'white', 'relief': tk.RAISED, 'font': ('Arial', 10), 'padx': 10, 'pady': 5}).grid(row=0, column=0, padx=2, sticky="ew")
        tk.Button(button_frame, text="Auto-Load Latest Capture",
                  command=self.load_latest_capture, **{'bg': self.colors['accent2'], 'fg': 'white', 'relief': tk.RAISED, 'font': ('Arial', 10), 'padx': 10, 'pady': 5}).grid(row=0, column=1, padx=2, sticky="ew")
        tk.Button(button_frame, text="Generate Simplified DXF",
                  command=self.process_image, **{'bg': self.colors['accent2'], 'fg': 'white', 'relief': tk.RAISED, 'font': ('Arial', 10), 'padx': 10, 'pady': 5}).grid(row=0, column=2, padx=2, sticky="ew")

        # Status label
        self.status_label = tk.Label(self.scrollable_frame, text="", fg=self.colors['accent2'], font=('Arial', 10, 'italic'))
        self.status_label.grid(row=3, column=0, pady=2)

        # Add mousewheel scrolling
        self.main_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Initialize preview-related variables
        self.cap = None
        self.preview_running = False
        self.preview_thread = None
        self.frame_buffer = deque(maxlen=2)
        self.update_queue = Queue()
        
        # Start queue checking
        self.check_queue()
        
        # Bind resize event
        self.master.bind('<Configure>', self.on_window_resize)
        
        # Start preview
        self.open_live_preview()

        # Add to __init__ after other initializations
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    # === Application logic ===
    def get_resolution_tuple(self):
        return tuple(map(int, self.selected_resolution.get().split('x')))

    def change_resolution(self, selection):
        self.resolution_label.config(text=f"Capture resolution: {selection.replace('x', ' x ')}")
        self.open_live_preview()

    def change_camera(self, selection):
        """Modified camera change function"""
        self.selected_camera.set(selection)
        # Open preview with new camera without checking resolutions
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
            try:
                ret, frame = self.cap.read()
                if ret:
                    self.frame_buffer.append(frame.copy())
                    if self.frame_buffer:
                        frame_to_process = self.frame_buffer[-1].copy()
                        self.process_and_queue_gui_update(frame_to_process)
                time.sleep(0.03)
            except Exception as e:
                print(f"Error in buffered_preview: {e}")
                time.sleep(0.1)  # Add delay on error

    def process_and_queue_gui_update(self, frame):
        try:
            # Get dimensions through a thread-safe call
            self.master.after_idle(self._get_dimensions_and_process, frame.copy())
        except Exception as e:
            print(f"Error in process_and_queue_gui_update: {e}")

    def _get_dimensions_and_process(self, frame):
        """Process frame update in the main thread"""
        try:
            # Now we can safely get the canvas width
            canvas_width = self.main_canvas.winfo_width()
            if canvas_width < 10:  # Not yet properly initialized
                canvas_width = 900  # Default width
            
            # Calculate preview width (1/3 of canvas width for each preview)
            preview_width = max(100, (canvas_width // 3) - 10)
            
            # Calculate height maintaining aspect ratio
            aspect_ratio = frame.shape[0] / frame.shape[1]
            preview_height = int(preview_width * aspect_ratio)
            
            # Resize frame
            frame_resized = cv2.resize(frame, (preview_width, preview_height))

            # Initialize edges and mask_blend
            edges = None
            mask_blend = None

            if self.color_mode.get() and self.target_color is not None:
                # Get both mask and edges from color detection
                mask, edges = color_based_edge_detection(
                    frame_resized, 
                    self.target_color,
                    tolerance_h=self.color_tolerance_h.get(),
                    tolerance_s=self.color_tolerance_s.get(),
                    tolerance_v=self.color_tolerance_v.get()
                )
                
                # Create a colored visualization of the mask
                mask_colored = np.zeros_like(frame_resized)
                mask_colored[mask > 0] = [0, 255, 0]  # Green for detected areas
                
                # Blend with original image
                mask_blend = cv2.addWeighted(frame_resized, 0.7, mask_colored, 0.3, 0)
                
            else:
                # Standard edge detection
                gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, self.canny_low.get(), self.canny_high.get())
                mask_blend = np.zeros_like(frame_resized)  # Empty mask display for non-color mode

            # Ensure edges exists
            if edges is None:
                edges = np.zeros((preview_height, preview_width), dtype=np.uint8)

            # Convert edges to RGB for display
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

            # Update the GUI with all three images
            self.update_gui_from_main_thread(frame_resized, edges_rgb, mask_blend)

        except Exception as e:
            print(f"Error in _get_dimensions_and_process: {e}")
            # Create blank images in case of error
            blank = np.zeros((preview_height, preview_width, 3), dtype=np.uint8)
            self.update_gui_from_main_thread(frame_resized, blank, blank)

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
        try:
            # Handle frame
            if len(frame.shape) == 2:  # If grayscale
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:  # If already RGB/BGR
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_original = Image.fromarray(frame_rgb)
            imgtk_original = ImageTk.PhotoImage(image=img_original)
            
            # Handle edges
            if len(edges.shape) == 2:  # If grayscale
                edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            else:  # If already RGB/BGR
                edges_rgb = edges  # Already in RGB format
            img_edges = Image.fromarray(edges_rgb)
            imgtk_edges = ImageTk.PhotoImage(image=img_edges)
            
            # Handle mask
            if len(mask.shape) == 2:  # If grayscale
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            else:  # If already RGB/BGR
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            img_mask = Image.fromarray(mask_rgb)
            imgtk_mask = ImageTk.PhotoImage(image=img_mask)
            
            # Update the labels
            self.preview_label_original.imgtk = imgtk_original
            self.preview_label_original.configure(image=imgtk_original)
            self.preview_label_edges.imgtk = imgtk_edges
            self.preview_label_edges.configure(image=imgtk_edges)
            self.preview_label_mask.imgtk = imgtk_mask
            self.preview_label_mask.configure(image=imgtk_mask)
            
        except Exception as e:
            print(f"Error in update_gui_from_main_thread: {e}")
            # Create blank images in case of error
            blank = np.zeros((100, 100, 3), dtype=np.uint8)
            self.preview_label_original.configure(image='')
            self.preview_label_edges.configure(image='')
            self.preview_label_mask.configure(image='')

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
            
            # Initialize edges variable
            edges = None
            
            if self.color_mode.get() and self.target_color is not None:
                # Get both edges and mask from color detection
                edges, mask = color_based_edge_detection(
                    frame,
                    self.target_color,
                    tolerance_h=self.color_tolerance_h.get(),
                    tolerance_s=self.color_tolerance_s.get(),
                    tolerance_v=self.color_tolerance_v.get()
                )
                
                # Ensure we have thick, clear edges for DXF generation
                kernel = np.ones((3,3), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=2)
                
                # Use the mask for edge detection
                edges = cv2.Canny(mask, 50, 150)
                
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, self.canny_low.get(), self.canny_high.get())

            # Ensure we have edges
            if edges is None:
                raise Exception("Failed to generate edges from capture")

            # Save both the original edges and a thresholded version
            cv2.imwrite(edges_path, edges)
            
            # Save a debug image to see what we're working with
            debug_path = os.path.join(self.capture_directory, f"capture_{timestamp}_debug.jpg")
            if self.color_mode.get():
                cv2.imwrite(debug_path, mask)

            self.image_path = edges_path
            self.edge_image = edges_path

            self.status_label.config(text=f"Captured and saved: {edges_path}")

        except Exception as e:
            messagebox.showerror("Capture Error", str(e))
            print(f"Capture error: {str(e)}")  # Add debug print
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
            print(f"Original image shape: {image.shape}")
            
            if self.color_mode.get() and self.target_color is not None:
                edges = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
                
                # Minimal morphological operations to preserve detail
                kernel_close = np.ones((2,2), np.uint8)  # Smaller kernel
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
                
                # Very light dilation
                kernel_dilate = np.ones((2,2), np.uint8)
                edges = cv2.dilate(edges, kernel_dilate, iterations=1)
                
                _, thresh = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
                
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

            # Use CHAIN_APPROX_NONE for maximum points
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            print(f"Initial contours found: {len(contours)}")

            debug_img = np.zeros_like(image)
            cv2.drawContours(debug_img, contours, -1, (0,255,0), 1)
            cv2.imwrite("contours_debug.png", debug_img)

            doc = ezdxf.new()
            msp = doc.modelspace()

            valid_contours = 0
            for i, contour in enumerate(contours):
                # Reduced minimum area threshold
                area = cv2.contourArea(contour)
                if area < 2:  # Even smaller minimum area (was 3)
                    continue
                
                # Much more detailed tolerance based on contour size
                if area > 1000:
                    tolerance = 0.05  # Ultra-high detail for large contours
                elif area > 500:
                    tolerance = 0.08
                elif area > 100:
                    tolerance = 0.1
                elif area > 50:
                    tolerance = 0.15
                else:
                    tolerance = 0.2  # Still very detailed for small contours
                
                # Get initial simplified contour
                simplified = simplify_contour(contour, tolerance=tolerance)
                
                # Convert to points
                points = [(pt[0][0] * inches_per_pixel, pt[0][1] * inches_per_pixel) for pt in simplified]
                
                # Add interpolated points for smoother curves
                if len(points) > 2:
                    # Create final point list with interpolation for smoother curves
                    final_points = []
                    for i in range(len(points)):
                        current_point = points[i]
                        final_points.append(current_point)
                        
                        # Add interpolated points between vertices
                        if i < len(points) - 1:
                            next_point = points[(i + 1) % len(points)]
                            # Add multiple interpolated points between vertices
                            for t in range(1, 4):  # Add 3 points between vertices
                                t = t / 4.0
                                interpolated_x = current_point[0] * (1 - t) + next_point[0] * t
                                interpolated_y = current_point[1] * (1 - t) + next_point[1] * t
                                final_points.append((interpolated_x, interpolated_y))
                    
                    try:
                        # Add the highly detailed polyline
                        msp.add_lwpolyline(final_points, close=True)
                        valid_contours += 1
                        print(f"Added polyline {valid_contours}")
                        print(f"Original points: {len(contour)}")
                        print(f"Simplified points: {len(points)}")
                        print(f"Final points with interpolation: {len(final_points)}")
                    except Exception as e:
                        print(f"Failed to add polyline: {e}")

            print(f"Valid contours processed: {valid_contours}")

            output_path = filedialog.asksaveasfilename(defaultextension=".dxf", filetypes=[("DXF files", "*.dxf")])
            if output_path:
                doc.saveas(output_path)
                messagebox.showinfo("Success", f"DXF saved to: {output_path}")
                self.status_label.config(text=f"DXF export complete. {valid_contours} contours processed.")

        except Exception as e:
            print(f"Error in process_image: {str(e)}")
            messagebox.showerror("Processing Error", str(e))

    def pick_color(self):
        if not self.frame_buffer:
            messagebox.showerror("Error", "No image available")
            return

        # Get the latest frame and resize for preview
        frame = self.frame_buffer[-1].copy()
        height, width = frame.shape[:2]
        preview_width, preview_height = width // 2, height // 2
        preview_frame = cv2.resize(frame, (preview_width, preview_height))
        preview_image = Image.fromarray(cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(preview_image)

        # Create a Toplevel window for color picking
        picker_win = tk.Toplevel(self.master)
        picker_win.title("Pick Color")
        picker_win.resizable(False, False)

        canvas = tk.Canvas(picker_win, width=preview_width, height=preview_height)
        canvas.pack()
        canvas.imgtk = imgtk  # Keep a reference!
        canvas.create_image(0, 0, anchor="nw", image=imgtk)

        # Draw circle on mouse move
        radius = self.color_sample_radius.get()
        circle = None

        def on_mouse_move(event):
            nonlocal circle
            canvas.delete("circle")
            x, y = event.x, event.y
            circle = canvas.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                outline="green", width=2, tags="circle"
            )

        def on_mouse_click(event):
            # Map coordinates back to original frame
            orig_x = int(event.x * (width / preview_width))
            orig_y = int(event.y * (height / preview_height))
            orig_radius = int(radius * (width / preview_width))
            color = self.get_average_color(frame, orig_y, orig_x, orig_radius)
            self._update_color_selection(color)
            picker_win.destroy()

        canvas.bind("<Motion>", on_mouse_move)
        canvas.bind("<Button-1>", on_mouse_click)

        # Optional: close picker if main window closes
        picker_win.transient(self.master)
        picker_win.grab_set()
        self.master.wait_window(picker_win)

    def get_average_color(self, frame, center_y, center_x, radius):
        """Calculate average color in a circular region"""
        try:
            height, width = frame.shape[:2]
            
            # Create mask for circular region
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Ensure coordinates and radius are within bounds
            center_x = max(radius, min(center_x, width - radius))
            center_y = max(radius, min(center_y, height - radius))
            
            # Draw the circle for sampling
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            
            # Get colors within the circular mask
            colors = frame[mask == 255]
            
            # Calculate average color if any pixels are selected
            if len(colors) > 0:
                return np.mean(colors, axis=0).astype(np.uint8)
            return frame[center_y, center_x]  # Fallback to single pixel
            
        except Exception as e:
            print(f"Error in get_average_color: {e}")
            return frame[center_y, center_x]  # Fallback to single pixel

    def _update_color_selection(self, color):
        """Update color selection from main thread"""
        try:
            self.target_color = color
            self.color_mode.set(True)
            
            # Update color preview
            hex_color = '#{:02x}{:02x}{:02x}'.format(color[2], color[1], color[0])
            self.color_preview.configure(bg=hex_color)
            
            self.refresh_preview()
            
            # Print color values for debugging
            hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
            print(f"Selected color - BGR: {color}, HSV: {hsv_color}")
            
        except Exception as e:
            print(f"Error updating color selection: {e}")
            messagebox.showerror("Error", f"Failed to update color selection: {e}")

    def _on_mousewheel(self, event):
        self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def on_window_resize(self, event=None):
        if self.frame_buffer:
            self.refresh_preview()

    def on_canvas_configure(self, event):
        """Handle canvas resize event"""
        # Update the width of the scrollable frame to match canvas
        self.main_canvas.itemconfig(self.canvas_frame, width=event.width)
        
        # Update preview sizes if needed
        if hasattr(self, 'preview_frame'):
            # Calculate new preview width (1/3 of canvas width for each preview)
            new_width = max(100, event.width // 3 - 10)  # Minimum width of 100px
            if self.frame_buffer:
                self.refresh_preview()

    def on_closing(self):
        """Handle application closing"""
        self.preview_running = False
        if hasattr(self, 'color_picker_window'):
            try:
                cv2.destroyWindow(self.color_picker_window)
            except:
                pass
        if self.cap:
            self.cap.release()
        self.master.destroy()

    def get_camera_resolutions(self, camera_index):
        """Check available resolutions for the selected camera"""
        try:
            # Common resolutions to test
            test_resolutions = [
                (640, 480),    # VGA
                (800, 600),    # SVGA
                (1024, 768),   # XGA
                (1280, 720),   # HD
                (1280, 1024),  # SXGA
                (1920, 1080),  # Full HD
                (2560, 1440),  # QHD
                (3840, 2160)   # 4K
            ]
            
            supported_resolutions = []
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            
            if not cap.isOpened():
                print(f"Failed to open camera {camera_index}")
                return []

            for width, height in test_resolutions:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                # Read actual values
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # If we got a valid resolution and it's not already in our list
                if actual_width > 0 and actual_height > 0:
                    resolution = (actual_width, actual_height)
                    if resolution not in supported_resolutions:
                        supported_resolutions.append(resolution)
                        print(f"Supported resolution: {actual_width}x{actual_height}")

            cap.release()
            return supported_resolutions

        except Exception as e:
            print(f"Error checking resolutions: {e}")
            return []

    def update_resolution_menu(self, resolutions):
        """Update the resolution menu with the given resolutions"""
        if resolutions:
            menu = self.resolution_menu["menu"]
            menu.delete(0, "end")
            
            # Sort resolutions by total pixels (ascending)
            resolutions.sort(key=lambda x: x[0] * x[1])
            
            # Update resolution options
            resolution_strings = [f"{w}x{h}" for w, h in resolutions]
            for resolution in resolution_strings:
                menu.add_command(label=resolution,
                               command=lambda r=resolution: self.selected_resolution.set(r))
            
            # Set to highest available resolution by default
            self.selected_resolution.set(resolution_strings[-1])
            
            print(f"Available resolutions:")
            for res in resolution_strings:
                print(f"  {res}")

    def check_current_camera_resolutions(self):
        """Manual resolution check for current camera"""
        selection = self.selected_camera.get()
        target_index = None
        
        for index, name in self.camera_index_map.items():
            if name == selection:
                target_index = index
                break
        
        if target_index is not None:
            # Close the current preview before checking resolutions
            self.close_preview()
            
            self.status_label.config(text="Checking available resolutions...")
            self.master.update()  # Update UI
            
            resolutions = self.get_camera_resolutions(target_index)
            self.update_resolution_menu(resolutions)
            
            if resolutions:
                res_text = "Available resolutions:\n" + "\n".join(f"{w}x{h}" for w, h in resolutions)
                messagebox.showinfo("Camera Resolutions", res_text)
            else:
                messagebox.showwarning("Camera Resolutions", "Could not determine available resolutions")
            
            self.status_label.config(text="Resolution check complete")
            
            # Restart the live preview
            self.open_live_preview()
        else:
            messagebox.showerror("Error", "No camera selected")

# === Run the app ===
if __name__ == "__main__":
    root = tk.Tk()
    app = CNCVisionApp(root)
    root.mainloop()