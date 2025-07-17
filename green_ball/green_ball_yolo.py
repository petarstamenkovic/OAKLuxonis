# This code demonstrates how to use a custom YoloV8 model to detect a green ball.

# Import necessary libraries
import cv2
import numpy as np
import depthai as dai
from ultralytics import YOLO

# Load the custom model
model = YOLO("green_ball_dataset/runs/detect/train2/weights/best.pt")   

# Create pipeline
pipeline = dai.Pipeline()

# Create a camera node
cameraNode = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
cameraOut = cameraNode.requestOutput(size=(640, 640), fps=15)

# Create an output queue for the camera
imageOut = cameraOut.createOutputQueue()

# Start the pipeline
pipeline.start()

# Main loop to capture frames and perform detection
while pipeline.isRunning:
    rgbImage = imageOut.get()
    frame = rgbImage.getCvFrame()

    # Run the model on the captured frame
    results = model(frame, 
                    verbose=False, 
                    conf = 0.85, 
                    iou = 0.5,
                    max_det = 1)

    # Results[0] contains x1, y1, x2, y2 coordinates of the detected bounding box
    annotated_frame = results[0].plot()

    print("Results: ", results)
    cv2.imshow("Detection Result", annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()