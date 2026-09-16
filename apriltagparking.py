"""Import every package installed in this project's venv and report its version.

Run it after building .venv to confirm the environment is sound:

    source .venv/bin/activate
    python apriltagparking.py
"""

# --- the three the project is built on -------------------------------------
import legoeducation            # LEGO Education SPIKE (code.legoeducation.com)
import cv2                      # opencv-python / opencv-contrib-python
import mediapipe as mp          # pinned to 0.10.x: 1.x aborts on Apple Silicon

# the Tasks API the hand-tracking scripts actually use
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# --- pulled in as dependencies, but useful directly ------------------------
import numpy as np
import bleak                    # Bluetooth LE backend legoeducation talks over
import matplotlib
import PIL                      # pillow
import sounddevice
import absl
import flatbuffers
import certifi
import cffi
import dateutil
import six

if __name__ == "__main__":
    for name, mod in [
        ("legoeducation", legoeducation),
        ("cv2", cv2),
        ("mediapipe", mp),
        ("numpy", np),
        ("bleak", bleak),
        ("matplotlib", matplotlib),
        ("pillow", PIL),
        ("sounddevice", sounddevice),
        ("absl-py", absl),
        ("flatbuffers", flatbuffers),
        ("certifi", certifi),
        ("cffi", cffi),
        ("python-dateutil", dateutil),
        ("six", six),
    ]:
        print(f"{name:16} {getattr(mod, '__version__', '(no __version__)')}")

    # prove the MediaPipe Tasks entry points resolved, not just the top module
    print(f"{'mp.tasks':16} {mp_python.BaseOptions.__name__}, "
          f"{mp_vision.HandLandmarker.__name__}")
