import depthai as dai

print(dai.__version__)

with dai.Device() as device:
	print("Connected device info: ")
	print(device.getDeviceInfo())
