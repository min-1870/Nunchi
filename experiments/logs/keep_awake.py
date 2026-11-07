"""keeps Windows from idle-sleeping while it runs (SetThreadExecutionState, released when the process ends)"""
import ctypes, time
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
while True:
    time.sleep(60)
