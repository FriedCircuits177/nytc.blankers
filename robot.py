"""
robot.py

Contains the class definition for the Robot object.
"""
import pygame
import time
import importlib
import cv2
import numpy as np
from ugot import ugot
import pose_yolo
importlib.reload(pose_yolo)
from pose_yolo import run_pose_control_inline
from IPython.display import display, clear_output, Image
from PIL import Image as Image2
from pose_yolo import run_pose_control_inline
import threading

import exceptions

class Robot():
    def __init__(self,channels,ip='192.168.88.1'):
        self.channels = channels
        self.ugot = ugot.UGOT()

        #try-except block to hopefully catch any connection errors
        try:
            print("attempting to connect to ugot at:")
            self.ugot.initialize(ip)
        except:
            raise exceptions.InvalidUgotIP(f"Failed to connect to UGOT at IP {ip}.\nPlease check the WiFi connection and IP shown on the bot.")
        
        print("\nugot connected succesfully!")
        self.ugot.load_models(
            ["color_recognition", "word_recognition", "line_recognition", "apriltag_qrcode"]
        )
        self.ugot.open_camera()
        self.ugot.set_track_recognition_line(0)
        self.ugot.screen_display_background(0)

    def update_camera_frame(self):
        frame = self.ugot.read_camera_data()
        if frame is None:
            return None

        # 1. Decode to BGR (OpenCV Default)
        nparr = np.frombuffer(frame, dtype=np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None

        # 2. Prepare HSV for Detection Queue
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        if not self.channels.hsv_camera_frame_queue.full():
            self.channels.hsv_camera_frame_queue.put(img_hsv)

        # 3. Prepare RGB for Pygame Queue
        # We do the transpose here if Pygame needs (width, height) orientation
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pygame = np.transpose(img_rgb, (1, 0, 2)) 

        if not self.channels.camera_frame_queue.full():
            self.channels.camera_frame_queue.put(img_pygame)

    def mainloop(self):
        while True:
            self.update_camera_frame()