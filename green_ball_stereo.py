# This code demonstrates how to detect and track a green ball.

# Import necessary libraries
import cv2
import depthai as dai

# Create a pipeline
pipeline = dai.Pipeline()

# Create a Camera node
camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
cameraOut = camera.requestOutput(size=(640, 480), fps=30)

# Stereo Depth node
stereo = pipeline.create(dai.node.StereoDepth)

# Mono cameras for stereo depth
monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

# Mono outputs from cameras
monoLeftOutput = monoLeft.requestOutput(size=(640, 480))
monoRightOutput = monoRight.requestOutput(size=(640, 480))

# Linking camera outputs to Stereo node
monoLeftOutput.link(stereo.left)
monoRightOutput.link(stereo.right)

# Create spatial calculation node
spatialCalc = pipeline.create(dai.node.SpatialLocationCalculator)
spatialCalc.inputConfig.setWaitForMessage(False)
spatialCalc.inputDepth.setBlocking(False)

# Link stereo node to spatial calculation node
stereo.depth.link(spatialCalc.inputDepth)

# Create output queues for spatial and image data
spatialOut = spatialCalc.out.createOutputQueue()
imageOut = cameraOut.createOutputQueue()

# Create an input queue for spatial configuration
spatialConfigIn = spatialCalc.inputConfig.createInputQueue()

# Start the pipeline
pipeline.start()

# Main loop for processing
while pipeline.isRunning():
    # Get a RGB frame
    frame = imageOut.get().getCvFrame()

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for detecting green color
    lower_green = (40, 100, 100)
    upper_green = (80, 255, 255)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    max_area = 500

    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            largest_contour = c

    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Normalize ROI for spatial calculation
        norm_x1 = x / frame.shape[1]
        norm_y1 = y / frame.shape[0]
        norm_x2 = (x + w) / frame.shape[1]
        norm_y2 = (y + h) / frame.shape[0]

        # Create a ROI for spatial location calculation
        roi = dai.Rect(dai.Point2f(norm_x1, norm_y1), dai.Point2f(norm_x2, norm_y2))

        # Create a SpatialLocationCalculatorConfigData object to send and configure the ROI
        ROI = dai.SpatialLocationCalculatorConfigData()
        ROI.roi = roi
        ROI.depthThresholds.lowerThreshold = 10  # Minimum depth in mm
        ROI.depthThresholds.upperThreshold = 10000  # Maximum depth in mm

        # Create a SpatialLocationCalculatorConfig object and set the ROI
        config = dai.SpatialLocationCalculatorConfig()
        config.setROIs([ROI])

        # Send the configuration to the spatial calculator
        spatialConfigIn.send(config)

        # Get spatial data
        spatialData = spatialOut.get().getSpatialLocations()
        
        # Print spatial coordinates if available
        if spatialData:
            coords = spatialData[0].spatialCoordinates
            cv2.putText(frame, f"X: {coords.x:.2f} mm", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(frame, f"Y: {coords.y:.2f} mm", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(frame, f"Z: {coords.z:.2f} mm", (x, y),      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Draw a circle at the center of the image
    cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2), 5, (0, 0, 255), -1)


    # Show output
    cv2.imshow("Green Ball Tracker", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()