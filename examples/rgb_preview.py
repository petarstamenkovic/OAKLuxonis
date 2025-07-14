# This code demonstrates how to create a simple RGB preview using DepthAI's ColorCamera node. It captures video from the camera and displays it in a window.

# Imort necessary libraries
import depthai as dai
import cv2

# Create pipeline
pipeline = dai.Pipeline()

# Define nodes 
camera = pipeline.create(dai.node.Camera)
camera.build(boardSocket=dai.CameraBoardSocket.CAM_A)

# Set camera properties
cameraOut = camera.requestOutput(
    size=(640, 480), 
    fps=30
)

# Create output queue
outQueue = cameraOut.createOutputQueue()

# Start the pipeline
pipeline.start()

while pipeline.isRunning():
    # Get frame from the output queue
    frame = outQueue.get().getCvFrame()

    # Display the frame
    cv2.imshow("RGB Preview", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()