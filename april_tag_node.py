import depthai as dai
import cv2


# Create a pipeline
pipeline = dai.Pipeline()

## Create nodes 
# Create color camera node
camera = pipeline.create(dai.node.Camera)
camera.build(boardSocket=dai.CameraBoardSocket.CAM_A)


cameraOut = camera.requestOutput(
    size = (1920,1080), 
    fps=30)

# Create April tag node
aprilTagNode = pipeline.create(dai.node.AprilTag)
# Add parameters if needed

# Link camera preview output to AprilTag node input image
cameraOut.link(aprilTagNode.inputImage)

passthroughOutputQueue = aprilTagNode.passthroughInputImage.createOutputQueue()
outQueue = aprilTagNode.out.createOutputQueue()

pipeline.start()
while pipeline.isRunning():
    passthroughImage : dai.ImgFrame = passthroughOutputQueue.get()
    frame = passthroughImage.getCvFrame()
    cv2.imshow("Preview", frame)
    if cv2.waitKey(1) == ord('q'):
        print("Detected q key press!")
        break