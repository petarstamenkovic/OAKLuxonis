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
cameraOut = cameraNode.requestOutput(size=(640, 640), fps=10)

# Create mono cameras for depth estimation
leftMono = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
rightMono = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

# Create outputs for the mono cameras
leftOut = leftMono.requestOutput(size=(640, 640), fps=10)
rightOut = rightMono.requestOutput(size=(640, 640), fps=10)

# Create a stereo depth node
stereo = pipeline.create(dai.node.StereoDepth)

# Link mono cameras to the stereo depth node
leftOut.link(stereo.left)
rightOut.link(stereo.right)

# Create a Spatial Location Calculator node for depth estimation
spatialCalc = pipeline.create(dai.node.SpatialLocationCalculator)
spatialCalc.inputConfig.setWaitForMessage(True)  # Wait for input configuration 
spatialCalc.inputDepth.setBlocking(False)

# Link the stereo depth output to the spatial location calculator
stereo.depth.link(spatialCalc.inputDepth)

# Create an output queue for the camera
imageOut = cameraOut.createOutputQueue()
calcOut = spatialCalc.out.createOutputQueue()

# Create an input queue for the spatial location calculator
calcConfig = spatialCalc.inputConfig.createInputQueue()

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
    # Remove label from YOLO detection frame
    annotated_frame = results[0].plot(labels=False)

    # Get bounding box for the first detection
    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Create spatial ROI config
        config = dai.SpatialLocationCalculatorConfigData()
        config.roi = dai.Rect(dai.Point2f(x1 / frame.shape[1], y1 / frame.shape[0]),
                              dai.Point2f(x2 / frame.shape[1], y2 / frame.shape[0]))
        config.depthThresholds.lowerThreshold = 100  # in mm
        config.depthThresholds.upperThreshold = 10000
        spatialConfig = dai.SpatialLocationCalculatorConfig()
        spatialConfig.addROI(config)
        calcConfig.send(spatialConfig)

        # Get spatial result
        spatialData = calcOut.get()
        roi_datas = spatialData.getSpatialLocations()
        if len(roi_datas) > 0:
            coords = roi_datas[0].spatialCoordinates
            text_x = x2  # right corner x
            text_y = max(y1 - 40, 10)

        cv2.putText(annotated_frame, f"X: {int(coords.x)} mm", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.putText(annotated_frame, f"Y: {int(coords.y)} mm", (text_x, text_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.putText(annotated_frame, f"Z: {int(coords.z)} mm", (text_x, text_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Draw a dot in the center of the image
    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2
    cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)

    # Draw x and y axis of the image
    cv2.line(annotated_frame, (center_x, 0), (center_x, frame.shape[0]), (255, 0, 0), 1)
    cv2.line(annotated_frame, (0, center_y), (frame.shape[1], center_y), (255, 0, 0), 1)
        
    #print("Results: ", results)
    cv2.imshow("Detection Result", annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()