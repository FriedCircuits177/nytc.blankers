"""
definitions.py

Contains global definitions for custom exceptions used in this codebase
"""

from datetime import datetime
if __name__ == "__main__": a = datetime.now()
import pygame
import time
import importlib
import cv2
import numpy as np
from ugot import ugot
import pose_yolo
importlib.reload(pose_yolo)
from pose_yolo import run_pose_control_inline
#from IPython.display import display, clear_output, Image
from PIL import Image as Image2
from pose_yolo import run_pose_control_inline
import threading
from queue import Queue

class QueueChannels:
    def __init__(self):
        self.camera_frame_queue = Queue(1)
        self.hsv_camera_frame_queue = Queue(1)
        self.webcam_frame_queue = Queue(1)
        
        self.timer_value = datetime.fromtimestamp(0)-datetime.fromtimestamp(0)
        self.timer_running = False
        
        self.phase = 1

class InvalidUgotIP(Exception):
    pass

WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)

class STANDBY:
    pass
class PHASE1:
    pass
class PHASE2:
    pass
class PHASE3_DRIVER:
    pass
class PHASE3:
    pass