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
        
        # Create a main frame to hold everything
        main_frame = ttk.Frame(master)
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Create canvas and scrollbar
        self.main_canvas = tk.Canvas(main_frame)  # Keep as tk.Canvas for image display
        self.scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.main_canvas.yview)
        
        # Create the scrollable frame
        self.scrollable_frame = ttk.Frame(self.main_canvas)
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

    def create_control_buttons(self):
        """Create the control buttons at the bottom of the window"""
        button_frame = ttk.Frame(self.scrollable_frame)
        button_frame.grid(row=2, column=0, sticky="ew", pady=2)
        button_frame.grid_columnconfigure((0,1,2), weight=1)

        # Create buttons with ttk
        capture_button = ttk.Button(button_frame, text="Capture Image", command=self.capture_image)
        capture_button.grid(row=0, column=0, padx=5, pady=2, sticky="ew")

        process_button = ttk.Button(button_frame, text="Process Image", command=self.process_image)
        process_button.grid(row=0, column=1, padx=5, pady=2, sticky="ew")

        load_button = ttk.Button(button_frame, text="Load Latest Capture", command=self.load_latest_capture)
        load_button.grid(row=0, column=2, padx=5, pady=2, sticky="ew")

        # Add camera calibration button to the left column
        calibration_button = ttk.Button(self.left_column, text="Camera Calibration",
                  command=self.open_calibration_window)
        calibration_button.grid(row=2, column=0, sticky="ew", pady=2)

        # Status label
        self.status_label = ttk.Label(self.scrollable_frame, text="", font=('Arial', 10, 'italic'))
        self.status_label.grid(row=3, column=0, pady=2)

    def create_edge_detection_panel(self):
        """Create the edge detection settings panel"""
        canny_frame = ttk.LabelFrame(self.left_column, text="Edge Detection")
        canny_frame.grid(row=0, column=0, sticky="ew", pady=1)
        canny_frame.grid_columnconfigure(0, weight=1)

        # Canny thresholds
        ttk.Label(canny_frame, text="Canny Low:").grid(row=0, column=0, sticky="w")
        ttk.Scale(canny_frame, from_=0, to=255, orient="horizontal",
                 variable=self.canny_low).grid(row=1, column=0, sticky="ew")
        
        ttk.Label(canny_frame, text="Canny High:").grid(row=2, column=0, sticky="w")
        ttk.Scale(canny_frame, from_=0, to=255, orient="horizontal",
                 variable=self.canny_high).grid(row=3, column=0, sticky="ew")

        # Edge scale
        ttk.Label(canny_frame, text="Edge Scale:").grid(row=4, column=0, sticky="w")
        ttk.Scale(canny_frame, from_=1.0, to=2.0, orient="horizontal",
                 variable=self.edge_scale).grid(row=5, column=0, sticky="ew")

        # Inches per pixel
        ttk.Label(canny_frame, text="Inches per Pixel:").grid(row=6, column=0, sticky="w")
        ttk.Entry(canny_frame, textvariable=self.inches_per_pixel).grid(row=7, column=0, sticky="ew")

        # DXF rotation
        ttk.Label(canny_frame, text="DXF Rotation (degrees):").grid(row=8, column=0, sticky="w")
        ttk.Entry(canny_frame, textvariable=self.dxf_rotation).grid(row=9, column=0, sticky="ew")

    def create_camera_settings_panel(self):
        """Create the camera settings panel"""
        camera_frame = ttk.LabelFrame(self.left_column, text="Camera Settings")
        camera_frame.grid(row=1, column=0, sticky="ew", pady=1)
        camera_frame.grid_columnconfigure(0, weight=1)

        camera_grid = ttk.Frame(camera_frame)
        camera_grid.grid(row=0, column=0, sticky="ew")
        camera_grid.grid_columnconfigure(1, weight=1)

        ttk.Label(camera_grid, text="Camera:").grid(row=0, column=0, sticky="w")
        self.camera_menu = ttk.Combobox(camera_grid, textvariable=self.selected_camera, 
                                      values=self.available_cameras)
        self.camera_menu.grid(row=0, column=1, sticky="ew", padx=2)
        self.camera_menu.bind('<<ComboboxSelected>>', lambda e: self.change_camera())

        ttk.Label(camera_grid, text="Resolution:").grid(row=1, column=0, sticky="w")
        resolutions = ["640x480", "1280x720", "1920x1080", "2560x1440", "3840x2160"]
        self.resolution_menu = ttk.Combobox(camera_grid, textvariable=self.selected_resolution,
                                          values=resolutions)
        self.resolution_menu.grid(row=1, column=1, sticky="ew", padx=2)
        self.resolution_menu.bind('<<ComboboxSelected>>', lambda e: self.change_resolution())

        check_res_button = ttk.Button(camera_frame, text="Check Available Resolutions",
                                    command=self.check_current_camera_resolutions)
        check_res_button.grid(row=1, column=0, sticky="ew", pady=1)

    def create_color_detection_panel(self):
        """Create the color detection settings panel"""
        color_frame = ttk.LabelFrame(self.right_column, text="Color Detection")
        color_frame.grid(row=0, column=0, sticky="ew", pady=1)
        color_frame.grid_columnconfigure(0, weight=1)

        # Color mode toggle and picker
        mode_frame = ttk.Frame(color_frame)
        mode_frame.grid(row=0, column=0, sticky="ew", pady=1)
        mode_frame.grid_columnconfigure(1, weight=1)

        color_controls = ttk.Frame(mode_frame)
        color_controls.grid(row=0, column=0, columnspan=2, sticky="ew")
        color_controls.grid_columnconfigure(1, weight=1)

        ttk.Checkbutton(color_controls, text="Use Color",
                       variable=self.color_mode,
                       command=self.refresh_preview).grid(row=0, column=0, sticky="w")

        picker_frame = ttk.Frame(color_controls)
        picker_frame.grid(row=0, column=1, sticky="e")

        ttk.Label(picker_frame, text="R:").grid(row=0, column=0)
        ttk.Spinbox(picker_frame, from_=0, to=10, width=3,
                   textvariable=self.color_sample_radius).grid(row=0, column=1, padx=1)

        self.color_preview = tk.Canvas(picker_frame, width=20, height=20,
                                     relief='solid', bd=1, bg='#808080')
        self.color_preview.grid(row=0, column=2, padx=1)

        ttk.Button(picker_frame, text="Pick",
                  command=self.pick_color).grid(row=0, column=3)

        # Color tolerance controls
        tolerance_frame = ttk.Frame(color_frame)
        tolerance_frame.grid(row=1, column=0, sticky="ew")
        tolerance_frame.grid_columnconfigure(0, weight=1)

        for i, (label, var) in enumerate([
            ("Hue", self.color_tolerance_h),
            ("Sat", self.color_tolerance_s),
            ("Val", self.color_tolerance_v)
        ]):
            ttk.Label(tolerance_frame, text=label).grid(row=i*2, column=0, sticky="w")
            ttk.Scale(tolerance_frame, from_=0, to=90 if i==0 else 255, orient="horizontal",
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
                                       orient="horizontal",
                                       command=lambda *args: self.update_camera_settings())
        self.exposure_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Brightness slider
        brightness_frame = ttk.Frame(self.exposure_controls)
        brightness_frame.pack(fill=tk.X, pady=2)
        ttk.Label(brightness_frame, text="Brightness:").pack(side=tk.LEFT, padx=2)
        self.brightness_slider = ttk.Scale(brightness_frame,
                                         from_=0, to=255,
                                         variable=self.brightness_var,
                                         orient="horizontal",
                                         command=lambda *args: self.update_camera_settings())
        self.brightness_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Contrast slider
        contrast_frame = ttk.Frame(self.exposure_controls)
        contrast_frame.pack(fill=tk.X, pady=2)
        ttk.Label(contrast_frame, text="Contrast:").pack(side=tk.LEFT, padx=2)
        self.contrast_slider = ttk.Scale(contrast_frame,
                                       from_=0, to=255,
                                       variable=self.contrast_var,
                                       orient="horizontal",
                                       command=lambda *args: self.update_camera_settings())
        self.contrast_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Initially disable manual controls if auto exposure is on
        self.toggle_exposure_controls()

    def create_dxf_settings_panel(self):
        """Create the DXF settings panel"""
        dxf_frame = ttk.LabelFrame(self.dxf_column, text="DXF Settings")
        dxf_frame.grid(row=0, column=0, sticky="ew", pady=1)
        dxf_frame.grid_columnconfigure(0, weight=1)

        # Reference point settings
        ref_frame = ttk.Frame(dxf_frame)
        ref_frame.grid(row=2, column=0, sticky="ew", pady=2)
        ref_frame.grid_columnconfigure(0, weight=1)

        ttk.Checkbutton(ref_frame, text="Use Reference Point",
                       variable=self.use_reference_point).grid(row=0, column=0, sticky="w", pady=1)

        # Table coordinates
        coord_frame = ttk.Frame(ref_frame)
        coord_frame.grid(row=1, column=0, sticky="ew", pady=1)
        coord_frame.grid_columnconfigure((0,1), weight=1)

        ttk.Label(coord_frame, text="Table X (inches):").grid(row=0, column=0, sticky="w")
        ttk.Entry(coord_frame, textvariable=self.reference_table_x, width=6).grid(row=0, column=1, sticky="ew", padx=1)

        ttk.Label(coord_frame, text="Table Y (inches):").grid(row=1, column=0, sticky="w")
        ttk.Entry(coord_frame, textvariable=self.reference_table_y, width=6).grid(row=1, column=1, sticky="ew", padx=1)

        # Set reference point button
        ttk.Button(ref_frame, text="Set Reference Point",
                  command=self.set_reference_point).grid(row=2, column=0, sticky="ew", pady=1)

        # Table boundary box settings
        boundary_frame = ttk.Frame(dxf_frame)
        boundary_frame.grid(row=3, column=0, sticky="ew", pady=2)
        boundary_frame.grid_columnconfigure(0, weight=1)

        ttk.Checkbutton(boundary_frame, text="Add Table Boundary Box",
                       variable=self.add_table_boundary).grid(row=0, column=0, sticky="w", pady=1)

        # Table dimensions
        dim_frame = ttk.Frame(boundary_frame)
        dim_frame.grid(row=1, column=0, sticky="ew", pady=1)
        dim_frame.grid_columnconfigure((0,1), weight=1)

        ttk.Label(dim_frame, text="Width (inches):").grid(row=0, column=0, sticky="w")
        ttk.Entry(dim_frame, textvariable=self.table_width, width=6).grid(row=0, column=1, sticky="ew", padx=1)

        ttk.Label(dim_frame, text="Height (inches):").grid(row=1, column=0, sticky="w")
        ttk.Entry(dim_frame, textvariable=self.table_height, width=6).grid(row=1, column=1, sticky="ew", padx=1)

    def create_future_panel(self):
        """Create the future settings panel (placeholder)"""
        future_label = ttk.LabelFrame(self.future_column, text="Future Settings")
        future_label.grid(row=0, column=0, sticky="nsew", pady=1)

    def create_preview_frame(self):
        """Create the preview frame with two image displays"""
        self.preview_frame = ttk.Frame(self.scrollable_frame)
        self.preview_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.preview_frame.grid_columnconfigure((0,1), weight=1)

        self.preview_label_original = ttk.Label(self.preview_frame)
        self.preview_label_original.grid(row=0, column=0, padx=2, pady=2, sticky="nsew")
        
        self.preview_label_edges = ttk.Label(self.preview_frame)
        self.preview_label_edges.grid(row=0, column=1, padx=2, pady=2, sticky="nsew")

    def create_settings_frame(self):
        """Create the settings frame with all control panels"""
        settings_frame = ttk.Frame(self.scrollable_frame)
        settings_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        # Make sure the settings frame uses full width
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        settings_frame.grid_columnconfigure((0,1,2,3), weight=1, uniform='equal')  # Four columns now

        # Create four columns
        self.left_column = ttk.Frame(settings_frame)
        self.left_column.grid(row=0, column=0, sticky="nsew", padx=1)
        self.left_column.grid_columnconfigure(0, weight=1)

        self.right_column = ttk.Frame(settings_frame)
        self.right_column.grid(row=0, column=1, sticky="nsew", padx=1)
        self.right_column.grid_columnconfigure(0, weight=1)

        self.dxf_column = ttk.Frame(settings_frame)
        self.dxf_column.grid(row=0, column=2, sticky="nsew", padx=1)
        self.dxf_column.grid_columnconfigure(0, weight=1)

        self.future_column = ttk.Frame(settings_frame)
        self.future_column.grid(row=0, column=3, sticky="nsew", padx=1)
        self.future_column.grid_columnconfigure(0, weight=1)

        # Create the control panels
        self.create_edge_detection_panel()
        self.create_camera_settings_panel()
        self.create_color_detection_panel()
        self.create_exposure_controls()
        self.create_future_panel()
        self.create_dxf_settings_panel() 