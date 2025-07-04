import depthai as dai
import cv2

# Create pipeline
pipeline = dai.Pipeline()

# Define nodes 
cam = pipeline.createColorCamera()
cam.setPreviewSize(640,480)
cam.setInterleaved(False)

# Create output node
xout = pipeline.createXLinkOut()
xout.setStreamName("Video")

cam.preview.link(xout.input)

# Connect to device and start a pipeline
with dai.Device(pipeline) as device:
    # Get output queue
    videoQueue = device.getOutputQueue(name="Video", maxSize=4, blocking=False)

    while True: 
        # Get frame from the queue
        frame = videoQueue.get().getCvFrame()

        # Display the frame
        cv2.imshow("RGB Preview", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()