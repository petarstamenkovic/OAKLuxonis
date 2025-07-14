# This code demonstrates how to use Spatial Location Calculation node in a DepthAI pipeline.

# Import necessary libraries
import depthai as dai
import cv2

# Create a pipeline
pipeline = dai.Pipeline()

# Standard image convention coordinate system - top left is (0,0) and bottom right is (1,1)
topLeft = dai.Point2f(0.4, 0.4)
bottomRight = dai.Point2f(0.6, 0.6)

# Constants
color = (0, 255, 0)
font = cv2.FONT_HERSHEY_TRIPLEX
step = 0.05  # Step size for ROI movement

# Create mono cameras for left and right
monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

# Create a StereoDepth node
stereoNode = pipeline.create(dai.node.StereoDepth)

# Create a Spatial Location Calculation node
spatialCalculatorNode = pipeline.create(dai.node.SpatialLocationCalculator)

# Create output from the mono cameras
monoLeftOut = monoLeft.requestOutput(size=(640, 480))
monoRightOut = monoRight.requestOutput(size=(640, 480))

# Linking the cameras to the StereoDepth node
monoLeftOut.link(stereoNode.left)
monoRightOut.link(stereoNode.right) 

# Configure the StereoDepth node
stereoNode.setRectification(True)
stereoNode.setExtendedDisparity(True)

# Configure the Spatial Location Calculation node
config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 10     # Minimum depth in mm
config.depthThresholds.upperThreshold = 10000  # Maximum depth in mm - All other values will be ignored
calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
config.roi = dai.Rect(topLeft,bottomRight)

# Dont wait for new configuration messages - use declared configuration
spatialCalculatorNode.inputConfig.setWaitForMessage(False)

# Provide ROI information to the Spatial Location Calculation node - initially!
spatialCalculatorNode.initialConfig.addROI(config)

# Link the camera output to the Spatial Location Calculation node input
stereoNode.depth.link(spatialCalculatorNode.inputDepth)

# Create output queues
passthroughOutputQueue = spatialCalculatorNode.passthroughDepth.createOutputQueue()
outCalculatorDataQueue = spatialCalculatorNode.out.createOutputQueue()

# Create an input queue for configuration messages - once the ROI moves
inputConfigQueue = spatialCalculatorNode.inputConfig.createInputQueue()

# Start the pipeline
pipeline.start()
print("Pipeline started.")

# Main loop to process frames
while pipeline.isRunning():
    spatialData = outCalculatorDataQueue.get().getSpatialLocations() # Returns a list of SpatialLocationCalculatorData objects
                                                                     # Each contains ROI, spatial coordinates, and depth   

    # Retrieve a raw depth frame from the Stereo output
    passthroughFrame : dai.ImgFrame = passthroughOutputQueue.get()

    # 2D array with depth information in mm - not visually interesting
    frameDepth = passthroughFrame.getCvFrame() 

    # Processing the depth frame
    depthFrameColor = cv2.normalize(frameDepth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1) # Covert from mm to 8-bit grayscale
    depthFrameColor = cv2.equalizeHist(depthFrameColor) # Boosts visibility in low contrast areas
    depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET) # Maps intensity to color

    for depthData in spatialData:
        roi = depthData.config.roi
        roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
        xmin = int(roi.topLeft().x)
        ymin = int(roi.topLeft().y)
        xmax = int(roi.bottomRight().x)
        ymax = int(roi.bottomRight().y)

        cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, font, 2)

        cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), font, 0.5, color)
        cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), font, 0.5, color)
        cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), font, 0.5, color)

        cv2.imshow("Depth: ", depthFrameColor)

    key = cv2.waitKey(1)
    if key == ord('q'):
       print("Exiting...")
       break

    new_config = False
    # Handle WASD keys to move the ROI
    if key == ord('w'):
        topLeft.y -= step
        bottomRight.y -= step
        new_config = True
    elif key == ord('s'):
        topLeft.y += step
        bottomRight.y += step
        new_config = True
    elif key == ord('a'):
        topLeft.x -= step
        bottomRight.x -= step
        new_config = True
    elif key == ord('d'):
        topLeft.x += step
        bottomRight.x += step
        new_config = True

    if new_config:
        config.roi = dai.Rect(topLeft, bottomRight)
        config.calculationAlgorithm = calculationAlgorithm
        cfg = dai.SpatialLocationCalculatorConfig()
        cfg.addROI(config)
        inputConfigQueue.send(cfg)  # Send the new configuration to the Spatial Location Calculator
        new_config = False

# Clean up
cv2.destroyAllWindows()