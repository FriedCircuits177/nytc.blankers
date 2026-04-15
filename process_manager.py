"""
process_manager.py

Contains the class for the top-level Process Manager overseeing everything
"""

from datetime import datetime
a = datetime.now()
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
import json

#other modules
import robot,gui,timer
from definitions import *

class Manager:

    def __init__(self,ip="192.168.88.1",resolution=[1280,720]):
        self.channels = QueueChannels()

        with open("config.json",'r') as file:
            self.config = json.load(file)

        if self.config["enable_gui"] == True:
            self.gui = gui.GUI(self.channels,resolution)

        self.robot = robot.Robot(self.channels,ip,self.config)   
        self.timer = timer.Timer(self.channels)
        

    def mainloop(self):        
        self.robot_thread = threading.Thread(target=self.robot.mainloop)
        self.robot_thread.start()

        self.timer_thread = threading.Thread(target=self.timer.mainloop)
        self.timer_thread.start()

        if self.config["enable_gui"]==True:
            self.gui.mainloop()
        else:
            while True:
                try:
                    pass
                except KeyboardInterrupt:
                    break
            

        
