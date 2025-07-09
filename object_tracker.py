# This code demonstrates how to use an Object Tracker node - Tracking people on Camera

# Import libraries 
import cv2
import depthai as dai
import time

# Create a pipeline
pipeline = dai.Pipeline()

# Full frames or cropped ones
fullFrameTracking = False

# Define all Camera nodes
camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
stereo = pipeline.create(dai.node.StereoDepth)

# Outputs from cameras
monoLeftOutput = monoLeft.requestOutput(size=(640,480))
monoRightOutput = monoRight.requestOutput(size=(640,480))

# Linking camera outputs to Stereo node
monoLeftOutput.link(stereo.left)
monoRightOutput.link(stereo.right)

# Create and build Spatial Detection Network node
spatialDetectionNetwork = pipeline.create(dai.node.SpatialDetectionNetwork).build(camRgb,stereo,"yolov6-nano")

# Configure the Spatial Detection Network
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)
labelMap = spatialDetectionNetwork.getClasses()

# Create an Object Tracker node
objectTracker = pipeline.create(dai.node.ObjectTracker)

# Configure the Spatial Detection Network
objectTracker.setDetectionLabelsToTrack([0])                                           # Tracking for person, its ID is 0 in yolo
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)                # ZERO_TERM_COLOR_HISTOGRAM or ZERO_TERM_IMAGELESS or SHORT_TERM_IMAGELESS or SHORT_TERM_KFC
objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)  # SMALLEST_ID or UNIQUE_ID

# Create output queues
preview = objectTracker.passthroughTrackerFrame.createOutputQueue()
tracklets = objectTracker.out.createOutputQueue()

# Full Frame Tracking 
if fullFrameTracking:
    #camRgb.requestFullResolutionOutput().link(objectTracker.inputTrackerFrame)
    fullResCamOut = camRgb.requestFullResolutionOutput()
    fullResCamOut.link(objectTracker.inputDetectionFrame)
    fullResCamOut.link(objectTracker.inputTrackerFrame)
    objectTracker.inputTrackerFrame.setBlocking(False)
    objectTracker.inputTrackerFrame.setMaxSize(1)
else:
    spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)            # Regular unchanged frame from the Camera
spatialDetectionNetwork.out.link(objectTracker.inputDetections)                        # What objects did you find?

# Parameters
startTime = time.monotonic()
counter = 0
fps = 0
color = (255, 255, 255)

# Start the pipeline 
pipeline.start()

# Main loop for processing 
while pipeline.isRunning():
    imgFrame = preview.get()
    track = tracklets.get()

    assert isinstance(imgFrame, dai.ImgFrame)
    assert isinstance(track, dai.Tracklets)

    # Calculating the FPS
    counter+=1
    current_time = time.monotonic()
    if(current_time - startTime) > 1 : 
        fps = counter / (current_time - startTime)
        counter = 0
        startTime = current_time

    # Retrieve a frame and tracklets
    frame = imgFrame.getCvFrame()
    trackletsData = track.tracklets
    for t in trackletsData:
        roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
        x1 = int(roi.topLeft().x)
        x2 = int(roi.bottomRight().x)
        y1 = int(roi.topLeft().y)
        y2 = int(roi.bottomRight().y)

        # Extracting label from tracked data
        try:
            label = labelMap[t.label]
        except:
            label = t.label

        # Print out the label - person
        cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

        # Print out the ID of the tracked object - each person has unique ID
        cv2.putText(frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

        # Print out if the object is TRACKED or LOST
        cv2.putText(frame, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

        # Print out spatial coordinates of an object
        cv2.putText(frame, f"X: {int(t.spatialCoordinates.x)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.putText(frame, f"Y: {int(t.spatialCoordinates.y)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.putText(frame, f"z: {int(t.spatialCoordinates.z)} mm", (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

        # Show the Frame
        cv2.imshow("tracker", frame)

    if cv2.waitKey(1) == ord('q'):
        break