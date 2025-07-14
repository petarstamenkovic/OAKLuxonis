# This code demonstrates the usage of an IMU node in a DepthAI pipeline.

# Import necessary libraries
import depthai as dai
import cv2

# Create a pipeline
pipeline = dai.Pipeline()

# Create an IMU node
imu = pipeline.create(dai.node.IMU)

# Enable IMU sensors
imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW],400)
imu.setBatchReportThreshold(1)  # Send values as soon as one sample is readty - immediate reporting
imu.setMaxBatchReports(10)  # Set maximum batch reports to 10

# Create an output queue
imuOutputQueue = imu.out.createOutputQueue()

# Start the pipeline
pipeline.start()

# Main loop to process frames
while pipeline.isRunning():
    imuData = imuOutputQueue.get()
    imuPackets = imuData.packets # Multiple packets due to batching

    print("IMU Data:")
    for packet in imuPackets:
        acceleroVal = packet.acceleroMeter # m/s^2
        gyroVal = packet.gyroscope         # rad/s

        imuF = "{:.06f}"  # 6 decimal places for IMU values

        print(f"Accelerometer: x={imuF.format(acceleroVal.x)}, y={imuF.format(acceleroVal.y)}, z={imuF.format(acceleroVal.z)}")
        print(f"Gyroscope: x={imuF.format(gyroVal.x)}, y={imuF.format(gyroVal.y)}, z={imuF.format(gyroVal.z)}")


# Clean up
cv2.destroyAllWindows()