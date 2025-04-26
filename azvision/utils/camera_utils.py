import cv2
import subprocess
import os
from datetime import datetime

def list_ffmpeg_cameras():
    """List available cameras using FFmpeg"""
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
    """Build a map of camera indices to names"""
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

def get_camera_resolutions(camera_index):
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

def get_latest_image(directory):
    """Get the most recent image from a directory"""
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.jpg', '.png'))]
    if not files:
        return None
    return max(files, key=os.path.getctime)

def print_camera_parameters(cap):
    """Debug function to print camera parameters"""
    params = {
        'BRIGHTNESS': cv2.CAP_PROP_BRIGHTNESS,
        'CONTRAST': cv2.CAP_PROP_CONTRAST,
        'SATURATION': cv2.CAP_PROP_SATURATION,
        'EXPOSURE': cv2.CAP_PROP_EXPOSURE,
        'GAIN': cv2.CAP_PROP_GAIN,
        'AUTO_EXPOSURE': cv2.CAP_PROP_AUTO_EXPOSURE
    }
    
    print("\nCamera Parameters:")
    for name, param in params.items():
        value = cap.get(param)
        print(f"{name}: {value}") 