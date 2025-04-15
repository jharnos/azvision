import subprocess

def list_dshow_devices():
    print("=== DirectShow Devices ===")
    cmd = ['ffmpeg', '-list_devices', 'true', '-f', 'dshow', '-i', 'dummy']
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    print(result.stderr)

list_dshow_devices()
