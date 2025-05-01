import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import time
import threading
from collections import deque
from queue import Queue, Empty
import traceback
from datetime import datetime
import ezdxf

from calibration.calibration_window import CalibrationWindow
from utils.image_utils import color_based_edge_detection, simplify_contour, normalize_image_safe
from utils.camera_utils import list_ffmpeg_cameras, build_camera_index_map, get_latest_image, print_camera_parameters

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

        # Initialize all variables
        self.initialize_variables()
        
        # Create UI components
        self.create_preview_frame()
        self.create_settings_frame()
        self.create_control_buttons()
        
        # Add mousewheel scrolling
        self.main_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Start queue checking
        self.check_queue()
        
        # Bind resize event
        self.master.bind('<Configure>', self.on_window_resize)
        
        # Start preview
        self.open_live_preview()

        # Add to __init__ after other initializations
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def initialize_variables(self):
        """Initialize all application variables"""
        self.image_path = None
        self.edge_image = None
        self.selected_camera = tk.StringVar()
        self.selected_resolution = tk.StringVar(value="1920x1080")
        self.inches_per_pixel = tk.DoubleVar(value=0.0604)
        self.canny_low = tk.IntVar(value=50)
        self.canny_high = tk.IntVar(value=150)
        self.dxf_rotation = tk.DoubleVar(value=-0.25)  # Default rotation of -0.25 degrees clockwise
        
        # Edge detection resolution control
        self.edge_scale = tk.DoubleVar(value=1.0)  # 1.0 = full resolution, 2.0 = double resolution
        
        # Edge visualization color (BGR format)
        self.edge_color = [0, 255, 0]  # Default green
        
        # Background subtraction variables
        self.background_image = None
        self.background_edges = None
        self.use_background_subtraction = tk.BooleanVar(value=False)
        
        # Color detection variables
        self.color_mode = tk.BooleanVar(value=False)
        self.target_color = None
        self.color_tolerance_h = tk.IntVar(value=15)
        self.color_tolerance_s = tk.IntVar(value=100)
        self.color_tolerance_v = tk.IntVar(value=100)
        self.last_mask = None
        self.color_sample_radius = tk.IntVar(value=2)

        # Camera exposure control variables
        self.auto_exposure = tk.BooleanVar(value=True)
        self.exposure_var = tk.IntVar(value=-5)
        self.brightness_var = tk.IntVar(value=128)
        self.contrast_var = tk.IntVar(value=128)

        # Camera setup
        self.capture_directory = "captures"
        os.makedirs(self.capture_directory, exist_ok=True)

        self.camera_index_map = build_camera_index_map()
        self.available_cameras = list(self.camera_index_map.values())

        if not self.available_cameras:
            messagebox.showerror("Camera Error", "No cameras detected via FFmpeg.")
            self.master.destroy()
            return

        self.selected_camera.set(self.available_cameras[0])

        # Preview-related variables
        self.cap = None
        self.preview_running = False
        self.preview_thread = None
        self.frame_buffer = deque(maxlen=2)
        self.update_queue = Queue()
        
        # Calibration points
        self.calibration_points = []
        self.known_distance = tk.DoubleVar(value=1.0) 

    def create_preview_frame(self):
        """Create the preview frame with two image displays"""
        self.preview_frame = tk.Frame(self.scrollable_frame, bd=2, relief=tk.GROOVE, bg=self.colors['secondary'])
        self.preview_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.preview_frame.grid_columnconfigure((0,1), weight=1)  # Two columns instead of three

        self.preview_label_original = tk.Label(self.preview_frame)
        self.preview_label_original.grid(row=0, column=0, padx=2, pady=2, sticky="nsew")
        
        self.preview_label_edges = tk.Label(self.preview_frame)
        self.preview_label_edges.grid(row=0, column=1, padx=2, pady=2, sticky="nsew")

    def create_settings_frame(self):
        """Create the settings frame with all control panels"""
        settings_frame = tk.Frame(self.scrollable_frame, bd=2, relief=tk.GROOVE, bg=self.colors['secondary'])
        settings_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        # Make sure the settings frame uses full width
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        settings_frame.grid_columnconfigure((0,1,2), weight=1, uniform='equal')

        # Create three columns
        self.left_column = tk.Frame(settings_frame, bg=self.colors['main'])
        self.left_column.grid(row=0, column=0, sticky="nsew", padx=1)
        self.left_column.grid_columnconfigure(0, weight=1)

        self.right_column = tk.Frame(settings_frame, bg=self.colors['main'])
        self.right_column.grid(row=0, column=1, sticky="nsew", padx=1)
        self.right_column.grid_columnconfigure(0, weight=1)

        self.future_column = tk.Frame(settings_frame, bg=self.colors['main'])
        self.future_column.grid(row=0, column=2, sticky="nsew", padx=1)
        self.future_column.grid_columnconfigure(0, weight=1)

        # Create the control panels
        self.create_edge_detection_panel()
        self.create_camera_settings_panel()
        self.create_color_detection_panel()
        self.create_exposure_controls()
        self.create_future_panel()
        self.create_dxf_settings_panel()
        self.create_background_subtraction_panel()  # Add new panel

    def create_edge_detection_panel(self):
        """Create the edge detection settings panel"""
        canny_frame = tk.LabelFrame(self.left_column, text="Edge Detection", 
                                  **{'bg': self.colors['secondary'], 'fg': self.colors['text'], 
                                     'font': ('Arial', 8, 'bold')})
        canny_frame.grid(row=0, column=0, sticky="ew", pady=1)
        canny_frame.grid_columnconfigure(0, weight=1)

        # Basic settings in edge detection frame
        tk.Label(canny_frame, text="Inches per Pixel:", font=('Arial', 8)).grid(row=0, column=0, sticky="w")
        tk.Entry(canny_frame, textvariable=self.inches_per_pixel, width=10).grid(row=1, column=0, sticky="ew", padx=2)

        tk.Label(canny_frame, text="Edge Detection Resolution:", font=('Arial', 8)).grid(row=2, column=0, sticky="w")
        tk.Scale(canny_frame, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, 
                 variable=self.edge_scale,
                 command=lambda _: self.refresh_preview()).grid(row=3, column=0, sticky="ew")
        tk.Label(canny_frame, text="(1.0 = camera resolution, 2.0 = double resolution)", 
                font=('Arial', 7)).grid(row=4, column=0, sticky="w")

        tk.Label(canny_frame, text="Lower Threshold", font=('Arial', 8)).grid(row=5, column=0, sticky="w")
        tk.Scale(canny_frame, from_=0, to=255, orient=tk.HORIZONTAL, 
                 variable=self.canny_low,
                 command=lambda _: self.refresh_preview()).grid(row=6, column=0, sticky="ew")

        tk.Label(canny_frame, text="Upper Threshold", font=('Arial', 8)).grid(row=7, column=0, sticky="w")
        tk.Scale(canny_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                 variable=self.canny_high,
                 command=lambda _: self.refresh_preview()).grid(row=8, column=0, sticky="ew")

    def create_camera_settings_panel(self):
        """Create the camera settings panel"""
        camera_frame = tk.LabelFrame(self.left_column, text="Camera Settings", 
                                   **{'bg': self.colors['secondary'], 'fg': self.colors['text'], 
                                      'font': ('Arial', 8, 'bold')})
        camera_frame.grid(row=1, column=0, sticky="ew", pady=1)
        camera_frame.grid_columnconfigure(0, weight=1)

        camera_grid = tk.Frame(camera_frame, bg=self.colors['secondary'])
        camera_grid.grid(row=0, column=0, sticky="ew")
        camera_grid.grid_columnconfigure(1, weight=1)

        tk.Label(camera_grid, text="Camera:", font=('Arial', 8)).grid(row=0, column=0, sticky="w")
        self.camera_menu = tk.OptionMenu(camera_grid, self.selected_camera, *self.available_cameras, 
                                       command=self.change_camera)
        self.camera_menu.grid(row=0, column=1, sticky="ew", padx=2)
        self.camera_menu.configure(font=('Arial', 8))

        tk.Label(camera_grid, text="Resolution:", font=('Arial', 8)).grid(row=1, column=0, sticky="w")
        resolutions = ["640x480", "1280x720", "1920x1080", "2560x1440", "3840x2160"]
        self.resolution_menu = tk.OptionMenu(camera_grid, self.selected_resolution, *resolutions,
                                           command=self.change_resolution)
        self.resolution_menu.grid(row=1, column=1, sticky="ew", padx=2)
        self.resolution_menu.configure(font=('Arial', 8))

        check_res_button = tk.Button(camera_frame, text="Check Available Resolutions",
                                   command=self.check_current_camera_resolutions,
                                   font=('Arial', 8))
        check_res_button.grid(row=1, column=0, sticky="ew", pady=1)

    def create_color_detection_panel(self):
        """Create the color detection settings panel"""
        color_frame = tk.LabelFrame(self.right_column, text="Color Detection", 
                                  **{'bg': self.colors['secondary'], 'fg': self.colors['text'], 
                                     'font': ('Arial', 8, 'bold')})
        color_frame.grid(row=0, column=0, sticky="ew", pady=1)
        color_frame.grid_columnconfigure(0, weight=1)

        # Color mode toggle and picker
        mode_frame = tk.Frame(color_frame, bg=self.colors['secondary'])
        mode_frame.grid(row=0, column=0, sticky="ew", pady=1)
        mode_frame.grid_columnconfigure(1, weight=1)

        color_controls = tk.Frame(mode_frame, bg=self.colors['secondary'])
        color_controls.grid(row=0, column=0, columnspan=2, sticky="ew")
        color_controls.grid_columnconfigure(1, weight=1)

        tk.Checkbutton(color_controls, text="Use Color",
                       variable=self.color_mode,
                       command=self.refresh_preview,
                       font=('Arial', 8)).grid(row=0, column=0, sticky="w")

        picker_frame = tk.Frame(color_controls, bg=self.colors['secondary'])
        picker_frame.grid(row=0, column=1, sticky="e")

        tk.Label(picker_frame, text="R:", font=('Arial', 8)).grid(row=0, column=0)
        tk.Spinbox(picker_frame, from_=0, to=10, width=3,
                   textvariable=self.color_sample_radius,
                   font=('Arial', 8)).grid(row=0, column=1, padx=1)

        self.color_preview = tk.Canvas(picker_frame, width=20, height=20,
                                     relief='solid', bd=1, bg='#808080')
        self.color_preview.grid(row=0, column=2, padx=1)

        tk.Button(picker_frame, text="Pick",
                  command=self.pick_color,
                  font=('Arial', 8)).grid(row=0, column=3)

        # Color tolerance controls
        tolerance_frame = tk.Frame(color_frame, bg=self.colors['secondary'])
        tolerance_frame.grid(row=1, column=0, sticky="ew")
        tolerance_frame.grid_columnconfigure(0, weight=1)

        for i, (label, var) in enumerate([
            ("Hue", self.color_tolerance_h),
            ("Sat", self.color_tolerance_s),
            ("Val", self.color_tolerance_v)
        ]):
            tk.Label(tolerance_frame, text=label, font=('Arial', 8)).grid(row=i*2, column=0, sticky="w")
            tk.Scale(tolerance_frame, from_=0, to=90 if i==0 else 255, orient=tk.HORIZONTAL,
                     variable=var,
                     command=lambda _: self.refresh_preview()).grid(row=i*2+1, column=0, sticky="ew")

    def create_exposure_controls(self):
        """Create the camera exposure controls panel"""
        exposure_frame = ttk.LabelFrame(self.right_column, text="Camera Exposure")
        exposure_frame.grid(row=1, column=0, sticky="ew", pady=2)
        exposure_frame.grid_columnconfigure(0, weight=1)
        
        # Auto/Manual exposure toggle
        ttk.Checkbutton(exposure_frame, text="Auto Exposure", 
                        variable=self.auto_exposure,
                        command=self.toggle_exposure_controls).pack(padx=5, pady=2)
        
        # Manual exposure controls container
        self.exposure_controls = ttk.Frame(exposure_frame)
        self.exposure_controls.pack(fill=tk.X, padx=5, pady=2)
        
        # Exposure slider
        exposure_frame = ttk.Frame(self.exposure_controls)
        exposure_frame.pack(fill=tk.X, pady=2)
        ttk.Label(exposure_frame, text="Exposure:").pack(side=tk.LEFT, padx=2)
        self.exposure_slider = ttk.Scale(exposure_frame, 
                                       from_=-13, to=0,
                                       variable=self.exposure_var,
                                       orient=tk.HORIZONTAL,
                                       command=lambda *args: self.update_camera_settings())
        self.exposure_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Brightness slider
        brightness_frame = ttk.Frame(self.exposure_controls)
        brightness_frame.pack(fill=tk.X, pady=2)
        ttk.Label(brightness_frame, text="Brightness:").pack(side=tk.LEFT, padx=2)
        self.brightness_slider = ttk.Scale(brightness_frame,
                                         from_=0, to=255,
                                         variable=self.brightness_var,
                                         orient=tk.HORIZONTAL,
                                         command=lambda *args: self.update_camera_settings())
        self.brightness_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Contrast slider
        contrast_frame = ttk.Frame(self.exposure_controls)
        contrast_frame.pack(fill=tk.X, pady=2)
        ttk.Label(contrast_frame, text="Contrast:").pack(side=tk.LEFT, padx=2)
        self.contrast_slider = ttk.Scale(contrast_frame,
                                       from_=0, to=255,
                                       variable=self.contrast_var,
                                       orient=tk.HORIZONTAL,
                                       command=lambda *args: self.update_camera_settings())
        self.contrast_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Initially disable manual controls if auto exposure is on
        self.toggle_exposure_controls()

    def create_future_panel(self):
        """Create the future settings panel (placeholder)"""
        future_label = tk.LabelFrame(self.future_column, text="Future Settings", 
                                   **{'bg': self.colors['secondary'], 'fg': self.colors['text'], 
                                      'font': ('Arial', 8, 'bold')})
        future_label.grid(row=0, column=0, sticky="nsew", pady=1)

    def create_dxf_settings_panel(self):
        """Create the DXF settings panel"""
        dxf_frame = tk.LabelFrame(self.future_column, text="DXF Settings", 
                                **{'bg': self.colors['secondary'], 'fg': self.colors['text'], 
                                   'font': ('Arial', 8, 'bold')})
        dxf_frame.grid(row=0, column=0, sticky="ew", pady=1)
        dxf_frame.grid_columnconfigure(0, weight=1)

        # Rotation adjustment
        tk.Label(dxf_frame, text="DXF Rotation (degrees):", font=('Arial', 8)).grid(row=0, column=0, sticky="w")
        tk.Entry(dxf_frame, textvariable=self.dxf_rotation, width=10).grid(row=1, column=0, sticky="ew", padx=2)

    def create_background_subtraction_panel(self):
        """Create the background subtraction settings panel"""
        bg_frame = tk.LabelFrame(self.future_column, text="Background Subtraction", 
                               **{'bg': self.colors['secondary'], 'fg': self.colors['text'], 
                                  'font': ('Arial', 8, 'bold')})
        bg_frame.grid(row=1, column=0, sticky="ew", pady=1)
        bg_frame.grid_columnconfigure(0, weight=1)

        # Background capture button
        tk.Button(bg_frame, text="Capture Background",
                 command=self.capture_background,
                 **{'bg': self.colors['accent1'], 'fg': 'white', 'relief': tk.RAISED, 
                    'font': ('Arial', 8), 'padx': 5, 'pady': 2}).grid(row=0, column=0, sticky="ew", pady=2)

        # Toggle for background subtraction
        tk.Checkbutton(bg_frame, text="Use Background Subtraction",
                      variable=self.use_background_subtraction,
                      command=self.refresh_preview,
                      font=('Arial', 8)).grid(row=1, column=0, sticky="w", pady=2)

        # Edge color picker
        color_frame = tk.Frame(bg_frame, bg=self.colors['secondary'])
        color_frame.grid(row=2, column=0, sticky="ew", pady=2)
        color_frame.grid_columnconfigure(1, weight=1)

        tk.Label(color_frame, text="Edge Color:", font=('Arial', 8)).grid(row=0, column=0, sticky="w")
        
        # Create clickable color preview
        self.edge_color_preview = tk.Canvas(color_frame, width=20, height=20,
                                          relief='solid', bd=1, bg='#00FF00',
                                          cursor="hand2")  # Add hand cursor
        self.edge_color_preview.grid(row=0, column=1, padx=5, sticky="w")
        self.edge_color_preview.bind("<Button-1>", lambda e: self.pick_edge_color())

    def create_control_buttons(self):
        """Create the control buttons at the bottom of the window"""
        button_frame = tk.Frame(self.scrollable_frame, bg=self.colors['main'])
        button_frame.grid(row=2, column=0, sticky="ew", pady=2)
        button_frame.grid_columnconfigure((0,1,2,3), weight=1)  # Four columns for four buttons
        
        tk.Button(button_frame, text="Capture Latest Image",
                  command=self.capture_image, 
                  **{'bg': self.colors['accent2'], 'fg': 'white', 'relief': tk.RAISED, 
                     'font': ('Arial', 10), 'padx': 10, 'pady': 5}).grid(row=0, column=0, padx=2, sticky="ew")

        tk.Button(button_frame, text="Auto-Load Latest Capture",
                  command=self.load_latest_capture, 
                  **{'bg': self.colors['accent2'], 'fg': 'white', 'relief': tk.RAISED, 
                     'font': ('Arial', 10), 'padx': 10, 'pady': 5}).grid(row=0, column=1, padx=2, sticky="ew")

        tk.Button(button_frame, text="Camera Calibration",
                  command=self.open_calibration_window,
                  **{'bg': self.colors['accent1'], 'fg': 'white', 'relief': tk.RAISED, 
                     'font': ('Arial', 10), 'padx': 10, 'pady': 5}).grid(row=0, column=2, padx=2, sticky="ew")

        tk.Button(button_frame, text="Generate Simplified DXF",
                  command=self.process_image, 
                  **{'bg': self.colors['accent2'], 'fg': 'white', 'relief': tk.RAISED, 
                     'font': ('Arial', 10), 'padx': 10, 'pady': 5}).grid(row=0, column=3, padx=2, sticky="ew")

        # Status label
        self.status_label = tk.Label(self.scrollable_frame, text="", fg=self.colors['accent2'], font=('Arial', 10, 'italic'))
        self.status_label.grid(row=3, column=0, pady=2) 

    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def on_window_resize(self, event=None):
        """Handle window resize events"""
        if self.frame_buffer:
            self.refresh_preview()

    def on_canvas_configure(self, event):
        """Handle canvas resize event"""
        # Update the width of the scrollable frame to match canvas
        self.main_canvas.itemconfig(self.canvas_frame, width=event.width)
        
        # Update preview sizes if needed
        if hasattr(self, 'preview_frame'):
            # Calculate new preview width (1/2 of canvas width for each preview)
            new_width = max(100, event.width // 2 - 10)  # Minimum width of 100px
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

    def get_resolution_tuple(self):
        """Convert resolution string to tuple of integers"""
        try:
            width, height = map(int, self.selected_resolution.get().split('x'))
            return (width, height)
        except:
            return (1920, 1080)  # Default to 1080p if parsing fails

    def change_resolution(self, selection):
        """Handle resolution change"""
        try:
            width, height = map(int, selection.split('x'))
            if self.cap is not None and self.cap.isOpened():
                # Set new resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                # Verify the resolution was set
                actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
                print(f"Resolution change requested: {width}x{height}")
                print(f"Actual camera resolution: {actual_width}x{actual_height}")
                
                # Restart preview with new resolution
                self.open_live_preview()
        except Exception as e:
            print(f"Error changing resolution: {e}")
            messagebox.showerror("Resolution Error", f"Failed to change resolution: {e}")

    def change_camera(self, selection):
        """Handle camera change"""
        self.selected_camera.set(selection)
        # Open preview with new camera without checking resolutions
        self.open_live_preview()

    def open_live_preview(self):
        """Open live camera preview"""
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
                # Get the selected resolution
                width, height = self.get_resolution_tuple()
                
                # Set the resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                # Verify the resolution was set
                actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
                print(f"Opening camera with resolution: {width}x{height}")
                print(f"Actual camera resolution: {actual_width}x{actual_height}")
                
                self.cap = cap
                
                # Apply initial camera settings
                self.update_camera_settings()
                
                self.preview_running = True
                self.preview_thread = threading.Thread(target=self.buffered_preview)
                self.preview_thread.daemon = True
                self.preview_thread.start()
                self.status_label.config(text=f"Live preview: {target_camera_name} ({actual_width}x{actual_height})")
            else:
                self.status_label.config(text=f"Failed to open live preview for: {target_camera_name}")
        else:
            self.status_label.config(text=f"No matching index for: {target_camera_name}")

    def close_preview(self):
        """Close the camera preview"""
        self.preview_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def buffered_preview(self):
        """Run the buffered preview loop"""
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
        """Process frame and queue GUI update"""
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
            
            # Calculate preview width (1/2 of canvas width for each preview)
            preview_width = max(100, (canvas_width // 2) - 10)
            
            # Calculate height maintaining aspect ratio
            aspect_ratio = frame.shape[0] / frame.shape[1]
            preview_height = int(preview_width * aspect_ratio)
            
            # Resize frame for preview
            frame_resized = cv2.resize(frame, (preview_width, preview_height))

            # Process edges at higher resolution if needed
            if self.color_mode.get() and self.target_color is not None:
                # Process at higher resolution if scale > 1.0
                if self.edge_scale.get() > 1.0:
                    scale = self.edge_scale.get()
                    h, w = frame.shape[:2]
                    frame_highres = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    edges, mask = color_based_edge_detection(
                        frame_highres,
                        self.target_color,
                        tolerance_h=self.color_tolerance_h.get(),
                        tolerance_s=self.color_tolerance_s.get(),
                        tolerance_v=self.color_tolerance_v.get(),
                        debug=False
                    )
                    # Scale down the edges for preview
                    edges = cv2.resize(edges, (preview_width, preview_height), 
                                     interpolation=cv2.INTER_AREA)
                    mask = cv2.resize(mask, (preview_width, preview_height), 
                                    interpolation=cv2.INTER_AREA)
                else:
                    edges, mask = color_based_edge_detection(
                        frame_resized, 
                        self.target_color,
                        tolerance_h=self.color_tolerance_h.get(),
                        tolerance_s=self.color_tolerance_s.get(),
                        tolerance_v=self.color_tolerance_v.get(),
                        debug=False
                    )
                
                # Create visualization with original frame
                edges_colored = frame_resized.copy()
                # Add edges with selected color
                edges_colored[edges > 0] = self.edge_color
                # Add semi-transparent color mask
                mask_colored = np.zeros_like(frame_resized)
                mask_colored[mask > 0] = [0, 0, 255]  # Red for color mask
                edges_colored = cv2.addWeighted(edges_colored, 1.0, mask_colored, 0.3, 0)
                edges = edges_colored
                
            else:
                # Process at higher resolution if scale > 1.0
                if self.edge_scale.get() > 1.0:
                    # Scale up the frame for edge detection
                    scale = self.edge_scale.get()
                    h, w = frame.shape[:2]
                    frame_highres = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    gray = cv2.cvtColor(frame_highres, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    edges = cv2.Canny(blurred, self.canny_low.get(), self.canny_high.get())
                    
                    # Scale down the edges for preview
                    edges = cv2.resize(edges, (preview_width, preview_height), 
                                     interpolation=cv2.INTER_AREA)
                    
                    # Create visualization with original frame
                    edges_colored = frame_resized.copy()
                    edges_colored[edges > 0] = self.edge_color  # Use selected color
                    edges = edges_colored
                else:
                    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    edges = cv2.Canny(blurred, self.canny_low.get(), self.canny_high.get())
                    # Convert edges to colored visualization
                    edges_colored = frame_resized.copy()
                    edges_colored[edges > 0] = self.edge_color  # Use selected color
                    edges = edges_colored

            # Add scale indicator if calibrated
            if hasattr(self, 'calibration_points') and len(self.calibration_points) == 2:
                cv2.line(frame_resized, 
                        (int(self.calibration_points[0][0] * preview_width/w), 
                         int(self.calibration_points[0][1] * preview_height/h)),
                        (int(self.calibration_points[1][0] * preview_width/w), 
                         int(self.calibration_points[1][1] * preview_height/h)),
                        (0, 255, 0), 2)
                
                # Add distance label
                mid_x = (self.calibration_points[0][0] + self.calibration_points[1][0]) // 2
                mid_y = (self.calibration_points[0][1] + self.calibration_points[1][1]) // 2
                cv2.putText(frame_resized, 
                           f"{self.known_distance.get():.2f}\"",
                           (int(mid_x * preview_width/w), 
                            int(mid_y * preview_height/h)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Update the GUI
            self.update_gui_from_main_thread(frame_resized, edges, frame_resized)

        except Exception as e:
            print(f"Error in _get_dimensions_and_process: {e}")

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

    def update_gui_from_main_thread(self, frame, edges, _):
        """Update GUI elements from the main thread"""
        try:
            # Handle frame - ensure proper color space conversion
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_original = Image.fromarray(frame_rgb)
            imgtk_original = ImageTk.PhotoImage(image=img_original)
            
            # Handle edges - ensure proper color space conversion
            if len(edges.shape) == 2:  # If grayscale
                edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            else:  # If already RGB/BGR
                edges_rgb = cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img_edges = Image.fromarray(edges_rgb)
            imgtk_edges = ImageTk.PhotoImage(image=img_edges)
            
            # Update the labels
            self.preview_label_original.imgtk = imgtk_original
            self.preview_label_original.configure(image=imgtk_original)
            self.preview_label_edges.imgtk = imgtk_edges
            self.preview_label_edges.configure(image=imgtk_edges)
            
        except Exception as e:
            print(f"Error in update_gui_from_main_thread: {e}")
            # Create blank images in case of error
            blank = np.zeros((100, 100, 3), dtype=np.uint8)
            self.preview_label_original.configure(image='')
            self.preview_label_edges.configure(image='')

    def refresh_preview(self):
        """Refresh the preview display"""
        if self.frame_buffer:
            frame = self.frame_buffer[-1]
            self.process_and_queue_gui_update(frame)

    def toggle_exposure_controls(self):
        """Toggle exposure controls based on auto exposure setting"""
        state = 'disabled' if self.auto_exposure.get() else '!disabled'
        for widget in self.exposure_controls.winfo_children():
            for child in widget.winfo_children():
                if isinstance(child, ttk.Scale):
                    child.state([state])
        self.update_camera_settings()

    def update_camera_settings(self, *args):
        """Update camera settings"""
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            try:
                if self.auto_exposure.get():
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = auto
                else:
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure_var.get())
                
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness_var.get())
                self.cap.set(cv2.CAP_PROP_CONTRAST, self.contrast_var.get())
                
                # Debug output
                print(f"Camera settings updated:")
                print(f"Auto Exposure: {self.auto_exposure.get()}")
                print(f"Exposure: {self.exposure_var.get()}")
                print(f"Brightness: {self.brightness_var.get()}")
                print(f"Contrast: {self.contrast_var.get()}")
            except Exception as e:
                print(f"Error updating camera settings: {str(e)}")

    def open_calibration_window(self):
        """Open the calibration window"""
        if not self.frame_buffer:
            messagebox.showerror("Error", "Camera preview must be running")
            return
        
        CalibrationWindow(self.master, self.inches_per_pixel, self.on_calibration_complete)

    def on_calibration_complete(self, new_scale):
        """Handle calibration completion"""
        if new_scale is not None:
            self.status_label.config(text=f"Calibration updated: 1 pixel = {new_scale:.6f} inches") 

    def capture_image(self):
        """Capture an image from the camera"""
        if self.cap is None or not self.cap.isOpened():
            messagebox.showerror("Error", "Camera is not initialized")
            return False

        try:
            # Use current camera settings (they're already set from preview)
            time.sleep(0.5)  # Small delay to ensure settings are applied
            
            # Capture multiple frames for averaging
            num_frames = 10  # Number of frames to capture
            frames = []
            edges_list = []
            
            print(f"Capturing {num_frames} frames for averaging...")
            self.status_label.config(text=f"Capturing {num_frames} frames...")
            
            for i in range(num_frames):
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    raise Exception("Failed to capture frame")
                
                frames.append(frame.copy())
                
                # Process edges for this frame
                if self.color_mode.get() and self.target_color is not None:
                    edges, _ = color_based_edge_detection(
                        frame,
                        self.target_color,
                        tolerance_h=self.color_tolerance_h.get(),
                        tolerance_s=self.color_tolerance_s.get(),
                        tolerance_v=self.color_tolerance_v.get(),
                        debug=False
                    )
                else:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    edges = cv2.Canny(blurred, self.canny_low.get(), self.canny_high.get())
                
                edges_list.append(edges)
                time.sleep(0.05)  # Small delay between captures
            
            # Average the frames
            avg_frame = np.mean(frames, axis=0).astype(np.uint8)
            
            # Combine edges using bitwise OR
            combined_edges = np.zeros_like(edges_list[0])
            for edges in edges_list:
                combined_edges = cv2.bitwise_or(combined_edges, edges)
            
            # Save debug images
            cv2.imwrite("debug_capture_raw.png", avg_frame)
            cv2.imwrite("debug_combined_edges.png", combined_edges)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"captured_image_{timestamp}.png"
            
            # Save the averaged image
            cv2.imwrite(image_path, avg_frame)
            
            self.image_path = image_path
            self.status_label.config(text=f"Image captured: {image_path}")
            
            return True
            
        except Exception as e:
            print(f"Capture error: {str(e)}")
            traceback.print_exc()
            messagebox.showerror("Capture Error", str(e))
            return False

    def load_latest_capture(self):
        """Load the most recent captured image"""
        latest = get_latest_image(self.capture_directory)
        if latest:
            self.image_path = latest
            self.status_label.config(text=f"Loaded image: {latest}")
        else:
            messagebox.showwarning("Warning", "No images found.")

    def process_image(self):
        """Process the current image and generate DXF"""
        if not self.image_path:
            messagebox.showerror("Error", "No image loaded or captured.")
            return

        inches_per_pixel = self.inches_per_pixel.get()
        if inches_per_pixel <= 0:
            messagebox.showerror("Error", "Inches per pixel must be greater than 0.")
            return

        try:
            image = cv2.imread(self.image_path)
            print(f"\nDXF Export Debug:")
            print(f"Current inches_per_pixel: {inches_per_pixel:.6f}")
            print(f"Image dimensions: {image.shape[1]}x{image.shape[0]} pixels")
            
            # Normalize the image to improve contrast
            image_float = image.astype(np.float32)
            image_normalized = cv2.normalize(image_float, None, 0, 255, cv2.NORM_MINMAX)
            image = image_normalized.astype(np.uint8)
            
            # Process at higher resolution if scale > 1.0
            if self.edge_scale.get() > 1.0:
                scale = self.edge_scale.get()
                h, w = image.shape[:2]
                image = cv2.resize(image, (int(w * scale), int(h * scale)))
                print(f"Processing at {w * scale}x{h * scale} resolution")
            
            if self.color_mode.get() and self.target_color is not None:
                edges, mask = color_based_edge_detection(
                    image,
                    self.target_color,
                    tolerance_h=self.color_tolerance_h.get(),
                    tolerance_s=self.color_tolerance_s.get(),
                    tolerance_v=self.color_tolerance_v.get(),
                    debug=True
                )
                # Create binary threshold from mask
                thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
            else:
                # Normal Canny edge detection with improved contrast
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, self.canny_low.get(), self.canny_high.get())
                # Create binary threshold from edges
                thresh = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)[1]

            # Apply background subtraction if enabled
            if self.use_background_subtraction.get() and self.background_edges is not None:
                # Save original edges for debug
                cv2.imwrite("debug_original_edges.png", thresh)
                cv2.imwrite("debug_background_edges.png", self.background_edges)
                
                # Subtract background edges from current edges
                thresh = cv2.subtract(thresh, self.background_edges)
                
                # Ensure we don't have negative values
                thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY)[1]
                
                # Save debug image of subtracted result
                cv2.imwrite("debug_subtracted_edges.png", thresh)
                
                print("Background subtraction applied:")
                print(f"Original edges pixels: {np.count_nonzero(thresh)}")
                print(f"Background edges pixels: {np.count_nonzero(self.background_edges)}")
                print(f"Subtracted edges pixels: {np.count_nonzero(thresh)}")

            # Save debug images
            cv2.imwrite("thresh_debug.png", thresh)
            cv2.imwrite("edges_debug.png", edges)

            # Find contours with different methods to ensure we don't miss any
            contours_external, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_tree, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Combine contours from both methods
            all_contours = contours_external + contours_tree
            
            print(f"Contours found (External): {len(contours_external)}")
            print(f"Contours found (Tree): {len(contours_tree)}")
            print(f"Total unique contours: {len(all_contours)}")

            # Create new DXF document with inches as units
            doc = ezdxf.new(setup=True)
            # Set DXF units to inches
            doc.header['$INSUNITS'] = 1      # 1 = Inches
            doc.header['$LUNITS'] = 2        # 2 = Decimal
            doc.header['$MEASUREMENT'] = 1   # 1 = English (inches)
            msp = doc.modelspace()

            # Get image height for vertical flipping
            img_height = image.shape[0]
            valid_contours = 0
            
            # Get rotation angle in radians
            rotation_angle = np.radians(self.dxf_rotation.get())
            
            # Debug image to visualize contours
            debug_contours = image.copy()
            
            for i, contour in enumerate(all_contours):
                # Process larger contours with more detail
                area = cv2.contourArea(contour)
                if area < 1:  # Reduced minimum area to catch more edges
                    continue
                
                # Calculate tolerance based on contour size
                if area > 1000:
                    tolerance = 0.2  # More detail for large contours
                elif area > 100:
                    tolerance = 0.3
                else:
                    tolerance = 0.4  # More detail for small contours
                
                # Simplify while preserving more points
                simplified = simplify_contour(contour, tolerance=tolerance)
                
                # Debug: Print points for first few contours
                if i < 3:
                    print(f"\nContour {i} debug:")
                    print(f"Contour area: {area:.2f} pixels")
                    print(f"Number of points before simplification: {len(contour)}")
                    print(f"Number of points after simplification: {len(simplified)}")
                    
                    # Print first few points before and after scaling
                    print("\nFirst few points:")
                    for j in range(min(3, len(simplified))):
                        orig_x, orig_y = simplified[j][0]
                        scaled_x = orig_x * inches_per_pixel
                        scaled_y = (img_height - orig_y) * inches_per_pixel
                        print(f"Point {j}:")
                        print(f"  Original: ({orig_x:.2f}, {orig_y:.2f})")
                        print(f"  Scaled: ({scaled_x:.2f}, {scaled_y:.2f})")
                
                # Draw contour on debug image
                cv2.drawContours(debug_contours, [contour], -1, (0, 255, 0), 1)
                
                # Convert points, flip Y coordinates, and apply rotation
                points = []
                for pt in simplified:
                    x = pt[0][0] * inches_per_pixel
                    y = (img_height - pt[0][1]) * inches_per_pixel  # Flip Y coordinate
                    
                    # Apply rotation around the center of the image
                    center_x = image.shape[1] * inches_per_pixel / 2
                    center_y = image.shape[0] * inches_per_pixel / 2
                    
                    # Translate to origin, rotate, then translate back
                    x_centered = x - center_x
                    y_centered = y - center_y
                    
                    x_rotated = x_centered * np.cos(rotation_angle) - y_centered * np.sin(rotation_angle)
                    y_rotated = x_centered * np.sin(rotation_angle) + y_centered * np.cos(rotation_angle)
                    
                    x_final = x_rotated + center_x
                    y_final = y_rotated + center_y
                    
                    points.append((x_final, y_final))
                
                if len(points) > 2:
                    try:
                        msp.add_lwpolyline(points, close=True)
                        valid_contours += 1
                        print(f"Added polyline {valid_contours} with {len(points)} points")
                    except Exception as e:
                        print(f"Failed to add polyline: {e}")

            # Save debug image with contours
            cv2.imwrite("debug_contours.png", debug_contours)
            
            print(f"Valid contours processed: {valid_contours}")

            output_path = filedialog.asksaveasfilename(defaultextension=".dxf", 
                                                      filetypes=[("DXF files", "*.dxf")])
            if output_path:
                doc.saveas(output_path)
                messagebox.showinfo("Success", f"DXF saved to: {output_path}")
                self.status_label.config(text=f"DXF export complete. {valid_contours} contours processed.")

        except Exception as e:
            print(f"Error in process_image: {str(e)}")
            traceback.print_exc()
            messagebox.showerror("Processing Error", str(e))

    def pick_color(self):
        """Open color picker window"""
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

    def capture_background(self):
        """Capture and process the background image"""
        if self.cap is None or not self.cap.isOpened():
            messagebox.showerror("Error", "Camera is not initialized")
            return

        try:
            # Capture multiple frames for averaging
            num_frames = 10
            frames = []
            edges_list = []
            
            print(f"Capturing {num_frames} frames for background...")
            self.status_label.config(text=f"Capturing background...")
            
            for i in range(num_frames):
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    raise Exception("Failed to capture frame")
                
                frames.append(frame.copy())
                
                # Process edges for this frame
                if self.color_mode.get() and self.target_color is not None:
                    edges, _ = color_based_edge_detection(
                        frame,
                        self.target_color,
                        tolerance_h=self.color_tolerance_h.get(),
                        tolerance_s=self.color_tolerance_s.get(),
                        tolerance_v=self.color_tolerance_v.get(),
                        debug=False
                    )
                else:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    edges = cv2.Canny(blurred, self.canny_low.get(), self.canny_high.get())
                
                edges_list.append(edges)
                time.sleep(0.05)
            
            # Average the frames
            self.background_image = np.mean(frames, axis=0).astype(np.uint8)
            
            # Combine edges using bitwise OR
            self.background_edges = np.zeros_like(edges_list[0])
            for edges in edges_list:
                self.background_edges = cv2.bitwise_or(self.background_edges, edges)
            
            # Save debug images
            cv2.imwrite("debug_background.png", self.background_image)
            cv2.imwrite("debug_background_edges.png", self.background_edges)
            
            self.status_label.config(text="Background captured successfully")
            self.use_background_subtraction.set(True)
            self.refresh_preview()
            
        except Exception as e:
            print(f"Background capture error: {str(e)}")
            traceback.print_exc()
            messagebox.showerror("Background Capture Error", str(e))

    def pick_edge_color(self):
        """Open color picker for edge visualization color"""
        # Open the color chooser dialog
        color = colorchooser.askcolor(color='#00FF00', title="Choose Edge Color")
        
        if color[1]:  # If a color was selected (not cancelled)
            # Convert hex color to BGR for OpenCV
            hex_color = color[1].lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            self.edge_color = [rgb[2], rgb[1], rgb[0]]  # Convert RGB to BGR
            
            # Update preview
            self.edge_color_preview.configure(bg=color[1])
            self.refresh_preview()

    def check_current_camera_resolutions(self):
        """Check and display available resolutions for current camera"""
        if self.cap is not None and self.cap.isOpened():
            try:
                # Get current resolution
                actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
                # Get selected resolution
                selected_width, selected_height = self.get_resolution_tuple()
                
                messagebox.showinfo("Current Resolution", 
                                  f"Selected resolution: {selected_width}x{selected_height}\n"
                                  f"Actual camera resolution: {actual_width}x{actual_height}")
            except Exception as e:
                print(f"Error checking resolution: {e}")
                messagebox.showerror("Error", f"Failed to check resolution: {e}")
        else:
            messagebox.showwarning("Warning", "Camera must be initialized first") 