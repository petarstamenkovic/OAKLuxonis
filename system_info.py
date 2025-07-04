# This code shows the version of the DepthAI library and the connected device information.
import depthai as dai

# Create a device instance
device = dai.Device()
print(dai.__version__)
print("Connected device info: ")
print(device.getDeviceInfo())
