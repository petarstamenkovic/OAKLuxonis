import depthai as dai

devices = dai.Device.getAllAvailableDevices()
if devices:
    print("OAK device(s) connected:")
    for dev in devices:
        print(f"  {dev.name} - {dev.state}")
else:
    print("No OAK device found.")
