import cv2
import numpy as np
import ezdxf

# === Settings ===

# Scale: inches per pixel (you will need to calibrate this!)
# Example: if 100 pixels = 4 inches, then INCHES_PER_PIXEL = 4 / 100 = 0.04
INCHES_PER_PIXEL = 0.04  # 🔧 Adjust this after calibration

# Input edge image
INPUT_IMAGE = 'table_edges.jpg'

# Output DXF file
OUTPUT_DXF = 'table_contours_inches.dxf'

# === Step 1: Load the image ===
image = cv2.imread(INPUT_IMAGE)
if image is None:
    print(f"❌ Error: Cannot load image {INPUT_IMAGE}")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Optional: Threshold to make sure edges are sharp
_, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

# === Step 2: Find contours ===
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"✅ Found {len(contours)} contours!")

# === Step 3: Create DXF ===
doc = ezdxf.new()
msp = doc.modelspace()

for contour in contours:
    if len(contour) < 5:
        # Skip very small contours (noise)
        continue

    # Convert contour points to real-world units (inches)
    points = []
    for point in contour:
        x_pixel, y_pixel = point[0]
        x_real = x_pixel * INCHES_PER_PIXEL
        y_real = y_pixel * INCHES_PER_PIXEL
        points.append((x_real, y_real))

    # Add polyline to DXF (closed shape)
    msp.add_lwpolyline(points, close=True)

# Save the DXF file
doc.saveas(OUTPUT_DXF)
print(f"✅ DXF file saved as {OUTPUT_DXF}")
