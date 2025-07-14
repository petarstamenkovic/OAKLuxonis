# This code demonstrates how to use a pre-build NN model to detect objects

# Import necessary libraries
import depthai as dai
import cv2
import time
import numpy as np

# Create a pipeline
pipeline = dai.Pipeline()

# Create a Camera node - set specfications
camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A,  # Use CAM_A or CAM_B depending on your device 
                                                sensorResolution=(1920,1080),
                                                sensorFps=30,
                                                )   

# Create a Detection Network node - using build here we connect these two nodes, camera as an input!
detectionNetwork = pipeline.create(dai.node.DetectionNetwork).build(camera, dai.NNModelDescription("yolov6-nano"))

# Fetch all available classes - depends entirely on the model used
labelMap = detectionNetwork.getClasses()

# Create an outputs from the Detection Network
passThrough = detectionNetwork.passthrough.createOutputQueue()
outDetectionNetwork = detectionNetwork.out.createOutputQueue()

# Start the pipeline
pipeline.start()

# Variables for frame processing
frame = None
detections = []
startTime = time.monotonic() # For FPS calculation
counter = 0
color2 = (255, 255, 255)

# Function to normalize bounding box coordinates - from (0.0 - 1.0 neural network outputs) to pixel coordinates
def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])  # shape[0] is height, shape[1] is width
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def displayFrame(name, frame):
    
    frame_resized = cv2.resize(frame, (1280, 720))  # Resize frame to fit the window
    color = (255, 0, 0)
    for detection in detections:
        bbox = frameNorm(
            frame_resized,
            (detection.xmin, detection.ymin, detection.xmax, detection.ymax),
        )
        cv2.putText(
            frame_resized,
            labelMap[detection.label],
            (bbox[0] + 10, bbox[1] + 20),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            255
        )
        cv2.putText(
            frame_resized,
            f"{int(detection.confidence * 100)}%",
            (bbox[0] + 10, bbox[1] + 40),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            255
        )
        cv2.rectangle(frame_resized, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.imshow(name, frame_resized)

# Main loop to process frames
while pipeline.isRunning():
    inRgb : dai.ImgFrame = passThrough.get()
    inDet : dai.ImgDetection = outDetectionNetwork.get()
    
    if inRgb is not None:
        frame = inRgb.getCvFrame()
        cv2.putText(
            frame,
            "NN fps {:.2f}".format(counter / (time.monotonic() - startTime)),
            (2,frame.shape[0] - 4),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.4,
            color2,
        )

    # Is some detection data available?
    if inDet is not None:
        detections = inDet.detections
        counter += 1

    # Does frame exist?
    if frame is not None:
        cv2.namedWindow("rgb", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("rgb", 1280, 720) # Customize the window size - uncomment if needed
        displayFrame("rgb", frame)
        print("FPS: {:.2f}".format(counter / (time.monotonic() - startTime)))
    
    # Check for key press
    if cv2.waitKey(1) == ord('q'):
        pipeline.stop()
        break


