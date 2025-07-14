# This code demonstrates how to use Edge Detector node in a DepthAI pipeline.

import depthai as dai
import cv2

# Create a pipeline
pipeline = dai.Pipeline()

# Create nodes 
camera = pipeline.create(dai.node.Camera)
camera.build(boardSocket=dai.CameraBoardSocket.CAM_A)

# Set camera properties
cameraOut = camera.requestOutput(
    size=(640, 480),
    fps=30
)

# Create Edge Detector node 
edgeDetector = pipeline.create(dai.node.EdgeDetector)

# Kernel configuration for edge detection
sobelHorizontalKernel = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]

sobelVerticalKernel = [
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
]

# Configure the Edge Detector node with Sobel filter kernels
edgeDetector.initialConfig.setSobelFilterKernels(sobelHorizontalKernel, sobelVerticalKernel)

# Link camera node output to Edge Detector node input
cameraOut.link(edgeDetector.inputImage)

# Create output queues
passthroughOutputQueue = edgeDetector.outputImage.createOutputQueue()

# Start the pipeline
pipeline.start()

while pipeline.isRunning():
    # Get the processed frame from the output queue
    passthroughImage: dai.ImgFrame = passthroughOutputQueue.get()
    frame = passthroughImage.getCvFrame()

    # Display the frame with edge detection applied
    cv2.imshow("Edge Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        print("Take care!")
        break

# Cleanup
cv2.destroyAllWindows()