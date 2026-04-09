"""
gui.py

Contains the class for the graphical user interface code
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

class GUI():
    def __init__(self,channels,resolution=[1280,720]):
        self.channels = channels
        self.resolution = resolution
        #ensure window is centered
        os.environ['SDL_VIDEO_CENTERED'] = '1'

        pygame.init()
        self.screen = pygame.display.set_mode(resolution)
        self.clock = pygame.time.Clock()
        self.camframe = pygame.Surface(resolution)

        pygame.display.set_caption("nytc.blank (alpha 0.0.1)")
    def convert_bot_camera_frame(self):
        if not self.channels.camera_frame_queue.empty():
            frame = self.channels.camera_frame_queue.get()
            # Frame is already (width, height, 3) from robot.py - correct format for pygame
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                self.camframe = pygame.surfarray.make_surface(frame)
    
    def mainloop(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            self.convert_bot_camera_frame()  # Update cached frame if new one available
            self.screen.blit(self.camframe, (0, 0))

            # Update display
            pygame.display.flip()
            self.clock.tick(30)  # 30 FPS
            
        pygame.quit()