"""Start a private software-rendered X server without Coppelia's library path."""
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

environment = dict(os.environ)
environment.pop('LD_LIBRARY_PATH', None)
environment.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
server = subprocess.Popen(['Xvfb', ':99', '-ac', '-screen', '0', '1280x1024x24'], env=environment)
try:
    for _ in range(100):
        if server.poll() is not None:raise RuntimeError('private X server failed')
        if Path('/tmp/.X11-unix/X99').exists():break
        time.sleep(.1)
    else:raise RuntimeError('private X server startup timed out')
    os.environ['DISPLAY'] = ':99'
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0,str(Path(script).resolve().parent))
    runpy.run_path(script, run_name='__main__')
finally:
    server.terminate()
    server.wait(timeout=10)
