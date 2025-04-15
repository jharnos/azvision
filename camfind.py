import cv2
cap = cv2.VideoCapture("video=16MP USB Camera", cv2.CAP_FFMPEG)
print(cap.isOpened())
