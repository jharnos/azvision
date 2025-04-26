"""Configuration settings for the AriZona Vision application"""

# Color scheme
COLORS = {
    'main': "#81d2c8",      # Main background
    'secondary': "#abe3d6",  # Secondary elements
    'accent1': "#e3c46e",    # Warm gold
    'accent2': "#e34f78",    # Rose
    'text': "#333333"        # Dark gray for text
}

# Default camera settings
DEFAULT_CAMERA_SETTINGS = {
    'resolution': "1920x1080",
    'inches_per_pixel': 0.0604,
    'canny_low': 50,
    'canny_high': 150,
    'color_tolerance_h': 15,
    'color_tolerance_s': 100,
    'color_tolerance_v': 100,
    'color_sample_radius': 2,
    'auto_exposure': True,
    'exposure': -5,
    'brightness': 128,
    'contrast': 128
}

# DXF export settings
DXF_SETTINGS = {
    'min_contour_area': 3,
    'large_contour_tolerance': 0.3,
    'medium_contour_tolerance': 0.4,
    'small_contour_tolerance': 0.5,
    'large_contour_threshold': 1000,
    'medium_contour_threshold': 100
}

# File paths
CAPTURE_DIRECTORY = "captures"
DEBUG_IMAGE_PREFIX = "debug_"
CAPTURED_IMAGE_PREFIX = "captured_image_"

# Preview settings
PREVIEW_BUFFER_SIZE = 2
PREVIEW_UPDATE_INTERVAL = 0.03  # seconds
PREVIEW_ERROR_DELAY = 0.1  # seconds

# GUI settings
GUI_SETTINGS = {
    'preview_min_width': 100,
    'preview_padding': 10,
    'button_padding': 10,
    'button_height': 5,
    'font_family': 'Arial',
    'font_size_small': 8,
    'font_size_normal': 10,
    'font_size_large': 12
} 