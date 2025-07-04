# This code demonstrates how to use the AprilTag node in a DepthAI pipeline.
# It captures video from a camera, processes it to detect AprilTags, and displays the results

# Import necessary libraries
import depthai as dai
import cv2
import numpy as np


# Create a pipeline
pipeline = dai.Pipeline()

## Create nodes 
# Create color camera node
camera = pipeline.create(dai.node.Camera)
camera.build(boardSocket=dai.CameraBoardSocket.CAM_A)

# Set camera properties
cameraOut = camera.requestOutput(
    size = (640,480), 
    fps=30)

# Create April tag node
aprilTagNode = pipeline.create(dai.node.AprilTag)
# Add parameters if needed

# Link camera preview output to AprilTag node input image
cameraOut.link(aprilTagNode.inputImage)

# Create output queues
passthroughOutputQueue = aprilTagNode.passthroughInputImage.createOutputQueue()
outQueue = aprilTagNode.out.createOutputQueue()

# Start the pipeline
pipeline.start()

# Main loop to proceqss frames
while pipeline.isRunning():
    passthroughImage : dai.ImgFrame = passthroughOutputQueue.get()
    frame = passthroughImage.getCvFrame()

    apriltTagData = outQueue.get()
    # If you detected an AprilTag, do the following
    if apriltTagData:
        print("AprilTag detected:", apriltTagData)
        # Draw green squares around detected AprilTags
        if hasattr(apriltTagData, 'aprilTags'):   # aprilTagData is like a dictionary variable and we are checking if it has 'aprilTags' key
            for tag in apriltTagData.aprilTags:
                # Extract corner coordinates
                pts = [
                    (int(tag.topLeft.x), int(tag.topLeft.y)),
                    (int(tag.topRight.x), int(tag.topRight.y)),
                    (int(tag.bottomRight.x), int(tag.bottomRight.y)),
                    (int(tag.bottomLeft.x), int(tag.bottomLeft.y))
                ]
                # Draw polygon on <frame> image
                cv2.polylines(frame, [np.array(pts)], isClosed=True, color=(0,255,0), thickness=2)


    cv2.imshow("Preview", frame)
    if cv2.waitKey(1) == ord('q'):
        print("Detected q key press!")
        break