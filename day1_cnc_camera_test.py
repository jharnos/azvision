import cv2

# Open the default camera (index 0)
camera_index = 0
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("❌ Error: Cannot open camera.")
    exit()

print("✅ Camera opened successfully!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Resize for faster display (optional)
    frame = cv2.resize(frame, (960, 720))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Show the original and edge-detected frames
    cv2.imshow("Original Feed", frame)
    cv2.imshow("Canny Edge Detection", edges)

    # Press 's' to save the frame
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("table_capture.jpg", frame)
        cv2.imwrite("table_edges.jpg", edges)
        print("✅ Image saved!")

    # Press 'q' to exit
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break


cap.release()
cv2.destroyAllWindows()
