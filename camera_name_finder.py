import wmi

c = wmi.WMI()
for camera in c.Win32_PnPEntity():
    if 'camera' in str(camera.Caption).lower():
        print(f"Camera found: {camera.Caption}")
