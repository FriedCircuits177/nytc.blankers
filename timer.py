"""
timer.py

Contains code for the Timer instance
"""

import pygame
import time
import importlib
import cv2
import numpy as np
import threading
import os
from datetime import datetime
from ugot import ugot
import pose_yolo
importlib.reload(pose_yolo)
from pose_yolo import run_pose_control_inline
from PIL import Image as Image2

class Timer():
    def __init__(self,channels):
        self.channels = channels
        self.start = None
    def start_running(self):
        self.start = datetime.now()
    def post_time_update(self):
        if self.start is not None:
            time_elapsed = datetime.now() - self.start
            self.channels.timer_value = time_elapsed
    def stop_running(self):
        self.start = None
    def mainloop(self):
        while True:
            if self.channels.timer_running:
                if self.start is None:
                    self.start_running()
                self.post_time_update()
            else:
                self.stop_running()
            time.sleep(0.01)  # 10ms update rate