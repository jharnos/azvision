import cv2
import numpy as np

def simplify_contour(contour, tolerance=0.1):
    """
    Simplify contour while preserving maximum detail
    tolerance: much lower value for very high detail
    """
    epsilon = tolerance * cv2.arcLength(contour, True) / 200.0  # Doubled precision (was 100.0)
    return cv2.approxPolyDP(contour, epsilon, True)

def normalize_image_safe(image):
    """Safely normalize an image while handling edge cases"""
    image_float = image.astype(np.float32)
    current_min = np.min(image_float)
    current_max = np.max(image_float)
    
    # If the image is uniform or nearly uniform, return the original
    if current_max - current_min < 1e-6:
        return image
        
    # Otherwise normalize
    image_normalized = (255 * (image_float - current_min) / (current_max - current_min))
    return image_normalized.clip(0, 255).astype(np.uint8)

def color_based_edge_detection(image, target_color, tolerance_h=30, tolerance_s=50, tolerance_v=50, debug=False):
    """
    Combined version with both numeric safety and previous improvements
    """
    try:
        # Normalize the image to match the target color range
        image_float = image.astype(np.float32)
        current_max = np.max(image_float)
        current_min = np.min(image_float)
        
        # Only normalize if the range is significantly different
        if current_max < 200:  # If image is darker than expected
            image_normalized = (255 * (image_float - current_min) / (current_max - current_min)).astype(np.uint8)
        else:
            image_normalized = image

        # Safe numeric handling
        target_color = np.array(target_color, dtype=np.int32)
        target_color_array = np.uint8([[target_color]])
        
        # Convert to HSV using normalized image
        hsv_target = cv2.cvtColor(target_color_array, cv2.COLOR_BGR2HSV)[0][0]
        hsv_image = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2HSV)
        
        # Safe numeric handling for HSV calculations
        h = np.int32(hsv_target[0])
        s = np.int32(hsv_target[1])
        v = np.int32(hsv_target[2])
        
        # Increased value tolerance
        tolerance_v = max(tolerance_v, 80)
        
        # Safe bounds calculations
        lower_h = int(max(0, h - tolerance_h))
        upper_h = int(min(180, h + tolerance_h))
        lower_s = int(max(0, s - tolerance_s))
        upper_s = int(min(255, s + tolerance_s))
        lower_v = int(max(0, v - tolerance_v))
        upper_v = int(min(255, v + tolerance_v))
        
        # Create bounds arrays
        lower_bound = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
        upper_bound = np.array([upper_h, upper_s, upper_v], dtype=np.uint8)
        
        # Create the mask
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        
        # Morphological operations
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Get edges from the mask
        edges = cv2.Canny(mask, 50, 150)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        if debug:
            # Save debug images
            cv2.imwrite('normalized_input_debug.png', image_normalized)
            cv2.imwrite('mask_debug.png', mask)
            debug_vis = image.copy()
            debug_vis[mask > 0] = [0, 255, 0]
            cv2.imwrite('color_detection_debug.png', debug_vis)
        
        return edges, mask
        
    except Exception as e:
        if debug:
            print(f"Error in color detection: {str(e)}")
        h, w = image.shape[:2]
        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8) 