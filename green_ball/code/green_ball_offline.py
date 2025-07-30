# Import necessary libraries
import cv2
import depthai as dai

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
cameraNode = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
monoLeftNode = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRightNode = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

# Create camera outputs
cameraOut = cameraNode.requestOutput(size=(640, 640), fps=30, type=dai.ImgFrame.Type.NV12)
monoLeftOut = monoLeftNode.requestOutput(size=(640, 640), fps=30, type=dai.ImgFrame.Type.NV12)
monoRightOut = monoRightNode.requestOutput(size=(640, 640), fps=30, type=dai.ImgFrame.Type.NV12)

# Define video encoders for each camera
ve1 = pipeline.create(dai.node.VideoEncoder)
ve2 = pipeline.create(dai.node.VideoEncoder)
ve3 = pipeline.create(dai.node.VideoEncoder)

# Create encoders, one for each camera, consuming the frames and encoding them using H.264 / H.265 encoding
ve1.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.H264_MAIN)
ve2.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.H265_MAIN)
ve3.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.H264_MAIN)

# Linking
monoLeftOut.link(ve1.input)
cameraOut.link(ve2.input)
monoRightOut.link(ve3.input)

# Create output queues for each video encoder and camera (for preview purposes)
ve1Out = ve1.out.createOutputQueue()
ve2Out = ve2.out.createOutputQueue()
ve3Out = ve3.out.createOutputQueue()
camOutQueue = cameraOut.createOutputQueue()

# Start pipeline
pipeline.start()

with open('mono1.h264', 'wb') as fileMono1H264, open('color.h264', 'wb') as fileColorH264, open('mono2.h264', 'wb') as fileMono2H264:
    print("Press Ctrl+C to stop encoding...")
    try:
        while pipeline.isRunning():
            frame = camOutQueue.get().getCvFrame()
            packetMonoLeft = ve1Out.get()
            packetCamera = ve2Out.get()
            packetMonoRight = ve3Out.get()
            if packetMonoLeft is not None:
                fileMono1H264.write(packetMonoLeft.getData())
            if packetCamera is not None:
                fileColorH264.write(packetCamera.getData())
            if packetMonoRight is not None:
                fileMono2H264.write(packetMonoRight.getData())
            cv2.imshow("Camera Preview", frame)
            if cv2.waitKey(1) == ord('q'):
                break
    except KeyboardInterrupt:
        pass

cv2.destroyAllWindows()

print("To view the encoded data, convert the stream file (.h264/.h265) into a video file (.mp4), using commands below:")
cmd = "ffmpeg -framerate 30 -i {} -c copy {}"
print(cmd.format("mono1.h264", "mono1.mp4"))
print(cmd.format("mono2.h264", "mono2.mp4"))
print(cmd.format("color.h264", "color.mp4"))