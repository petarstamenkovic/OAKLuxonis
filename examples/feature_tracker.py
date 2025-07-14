# This code demonstrates how to use a Feature Tracker 

# Import necessary libraries
import cv2
import depthai as dai

# Create a pipeline
pipeline = dai.Pipeline()

# Create a Camera node
camera = pipeline.create(dai.node.Camera).build(
    dai.CameraBoardSocket.CAM_A)

camOut = camera.requestOutput(size=(640,480),fps=30)

# Create a Image manipulation node
imageManip = pipeline.create(dai.node.ImageManip)
imageManip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)

# Link Camera and ImageManip node
camOut.link(imageManip.inputImage)

# Create and configure a Feature Track node
tracker = pipeline.create(dai.node.FeatureTracker)
tracker.initialConfig.setNumTargetFeatures(100)

# Create Corner detector 
corner = dai.FeatureTrackerConfig.CornerDetector()
corner.type = dai.FeatureTrackerConfig.CornerDetector.Type.HARRIS
corner.numMaxFeatures = 100
corner.numTargetFeatures = 100

# Configure the Feature Tracker with the corner detector
tracker.initialConfig.setCornerDetector(corner)

# Link Image Manip and Feature tracker node
imageManip.out.link(tracker.inputImage)

# Create output queues 
frameQueue = tracker.passthroughInputImage.createOutputQueue()
featureQueue = tracker.outputFeatures.createOutputQueue()

# Start a pipeline
pipeline.start()

# Main while loop for processing
while pipeline.isRunning():
    frame = frameQueue.get().getCvFrame()
    feature = featureQueue.get().trackedFeatures

    for f in feature:
        pt = (int(f.position.x), int(f.position.y))
        cv2.circle(frame, pt, 2, (0,255,0), -1)

    cv2.imshow("Tracked features", frame)

    if cv2.waitKey(1) == ord('q'):
        break
