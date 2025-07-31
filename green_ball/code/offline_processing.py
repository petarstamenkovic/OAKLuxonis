# This code detects a green ball and tracks it using a custom YOLO model.

# Import libraries 
import cv2
import numpy as np
import depthai as dai
from ultralytics import YOLO

# Load OAK D Lite video files - use absolute paths
#rgbVideo = cv2.VideoCapture("../offline_video_samples/color.mp4") # Relative path fails to work
rgbVideo = cv2.VideoCapture("/home/biosense-ms/Desktop/gitRepos/OAKLuxonis/green_ball/offline_video_samples/color.mp4")
monoLeftVideo = cv2.VideoCapture("/home/biosense-ms/Desktop/gitRepos/OAKLuxonis/green_ball/offline_video_samples/mono1.mp4")
monoRightVideo = cv2.VideoCapture("/home/biosense-ms/Desktop/gitRepos/OAKLuxonis/green_ball/offline_video_samples/mono2.mp4")

# Import the YOLO model
model = YOLO("green_ball_dataset/runs/detect/train2/weights/best.pt")

while True:
    ret, frame = rgbVideo.read()
    if not ret:
        print("End or unsuccessful read of RGB video.")
        break
    results = model(frame)
    
    # Process results
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow("RGB Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break


cv2.destroyAllWindows()