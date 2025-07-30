from ultralytics import YOLO
import cv2

model = YOLO("green_ball_dataset/runs/detect/train2/weights/best.pt")
results = model("green_ball_test_sample2.png", show=True)

# Display with OpenCV
annotated_frame = results[0].plot()
cv2.imshow("Detection Result", annotated_frame)

# Wait for key press to close
cv2.waitKey(0)
cv2.destroyAllWindows()