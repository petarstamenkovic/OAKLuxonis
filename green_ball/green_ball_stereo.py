# This code demonstrates how to detect green ball and show its spatial coordinates using HSV transformation.

# Import necessary libraries
import cv2
import depthai as dai
import numpy as np
from collections import deque 

# Create a pipeline
pipeline = dai.Pipeline()

# Create a Camera node
camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
cameraOut = camera.requestOutput(size=(640, 480), fps=30)

# Stereo Depth node
stereo = pipeline.create(dai.node.StereoDepth)

# Configure stereo depth for better stability
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)

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
spatialCalc.inputConfig.setWaitForMessage(True)
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

# Buffer to smooth coordinates - for spatial coordinates jittering
coord_buffer = deque(maxlen=15)

# Main loop for processing
while pipeline.isRunning():
    frame = imageOut.get().getCvFrame()

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for detecting green color and create a mask
    ### Adjustable values ###
    lower_green = (20, 100, 100)
    upper_green = (120, 255, 255)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Finds outlines of the green objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Contour parameters
    selected_contour = None
    max_area = 500

    # Find only one contour - later we can change it to more or all circles
    for c in contours:
        area = cv2.contourArea(c)
        if area < max_area:
            continue
        
        # Find the circle shape
        (x, y), radius = cv2.minEnclosingCircle(c)
        if radius <= 0:
            continue

        circle_area = np.pi * (radius ** 2)
        circularity = area / circle_area

        ### Adjustable circularity threshold ###
        if circularity > 0.6:  
            selected_contour = c

    # If a circle is found
    if selected_contour is not None:
        # Fetch the ROI for the selected contour
        x, y, w, h = cv2.boundingRect(selected_contour)

        # Add padding to ROI - bigger ROI for better spatial calculation
        # This section draws out a ROI on the image
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Normalize ROI for spatial calculation
        # This section calculates the ROI for the SpatialLocationCalculator
        norm_x1 = x1 / frame.shape[1]
        norm_y1 = y1 / frame.shape[0]
        norm_x2 = x2 / frame.shape[1]
        norm_y2 = y2 / frame.shape[0]

        roi = dai.Rect(dai.Point2f(norm_x1, norm_y1), dai.Point2f(norm_x2, norm_y2))

        # Create a new ROI configuration to match the data type for SpatialLocationCalculator 
        # This is done in order to set the depth thresholds, those are not available in SpatialLocationCalculatorConfig type
        ROI = dai.SpatialLocationCalculatorConfigData() 
        ROI.roi = roi
        ROI.depthThresholds.lowerThreshold = 10     # in mm
        ROI.depthThresholds.upperThreshold = 10000  # in mm

        # Create a ROI configuration and send it to the spatial calculator - 
        # Necessary since SpatialLocationCalculator requuires this type and not SpatialLocationCalculatorData
        config = dai.SpatialLocationCalculatorConfig()
        config.setROIs([ROI])
        spatialConfigIn.send(config)

        # Retrieve spatial data
        spatialData = spatialOut.get().getSpatialLocations()

        if spatialData:
            coords = spatialData[0].spatialCoordinates

            # Average the coordinates to smooth out jitter
            coord_buffer.append((coords.x, coords.y, coords.z))

            avg_x = sum(c[0] for c in coord_buffer) / len(coord_buffer)
            avg_y = sum(c[1] for c in coord_buffer) / len(coord_buffer)
            avg_z = sum(c[2] for c in coord_buffer) / len(coord_buffer)

            # Print out the coordinates on the frame
            cv2.putText(frame, f"X: {int(avg_x)} mm", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(frame, f"Y: {int(avg_y)} mm", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(frame, f"Z: {int(avg_z)} mm", (x, y),      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Draw a baseline crosshair at image center
    cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2), 5, (0, 0, 255), -1)
    cv2.line(frame, (frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[0]), (255, 0, 0), 1)
    cv2.line(frame, (0, frame.shape[0] // 2), (frame.shape[1], frame.shape[0] // 2), (255, 0, 0), 1)

    # Show output
    cv2.imshow("Green Ball Tracker", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
