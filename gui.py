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

from definitions import *

class GUI():
    def __init__(self,channels,resolution=[1280,720]):
        self.channels = channels
        self.resolution = resolution
        #ensure window is centered
        os.environ['SDL_VIDEO_CENTERED'] = '1'

        pygame.init()
        self.screen = pygame.display.set_mode(resolution)
        self.clock = pygame.time.Clock()
        self.botcamframe = pygame.Surface(resolution)
        self.webcamframe = pygame.Surface((320, 240))  # Default webcam preview size

        # Initialize webcam capture
        self.webcam = cv2.VideoCapture(0)
        if not self.webcam.isOpened():
            print("Warning: Webcam not available")
        else:
            # Set webcam properties for efficiency
            self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.webcam.set(cv2.CAP_PROP_FPS, 30)
            self.webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        pygame.display.set_caption("nytc.blank (alpha 0.0.1)")

        #import phase images
        self.phase_images = []
        for x in range(5):
            self.phase_images.append(pygame.image.load(f"assets/p{x}.png"))
        
        # Create red versions of phase images (swap grey to red)
        self.phase_images_red = []
        for image in self.phase_images:
            image_array = pygame.surfarray.array3d(image)
            image_array = np.transpose(image_array, (1, 0, 2))  # Adjust dimensions if needed
            image_red = image_array.copy()
            grey_mask = np.all(image_array == [128, 128, 128], axis=2)
            image_red[grey_mask] = [255, 0, 0]
            image_red = np.transpose(image_red, (1, 0, 2))
            red_surface = pygame.surfarray.make_surface(image_red)
            self.phase_images_red.append(red_surface)
        
        print("\npygame GUI succesfully initialised.\n")

    def convert_bot_camera_frame(self):
        if not self.channels.camera_frame_queue.empty():
            frame = self.channels.camera_frame_queue.get()
            
            # Pygame's make_surface is very fast if the array is already 
            # (width, height, 3). 
            self.botcamframe = pygame.surfarray.make_surface(frame)
    
    def convert_webcam_frame(self):
        # Non-blocking frame read: only process if a frame is ready in buffer
        ret, frame = self.webcam.read()
        if not ret:
            return  # Skip if no frame available
        
        # Queue raw frame for other consumers
        if self.channels.webcam_frame_queue.empty():
            self.channels.webcam_frame_queue.put(frame, block=False)
        
        # Convert BGR to RGB, horizontally flip, and transpose for pygame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_flipped = cv2.flip(frame_rgb, 1)  # Flip horizontally (1 = flip around vertical axis)
        frame_transposed = np.transpose(frame_flipped, (1, 0, 2))
        self.webcamframe = pygame.surfarray.make_surface(frame_transposed)

    def render_text(self,text,font="CollegiateOutlineFLF.ttf",size=50,colour=(255,255,255)):
        font = pygame.font.Font(f"assets/{font}",size=size)
        text_surface = font.render(text,True,colour)
        return text_surface
    def remzeroround(self,inpt,roundto=10,addzero=True):
        inptt = int(inpt) % roundto
        if addzero and inptt<10:
            text_input = f"0{inptt}"
        else:
            text_input = str(inptt)
        return text_input
    def drawBoundingBox(self,x1,y1,x2,y2,colour=(255,0,0),thickness=2,xoff=0,yoff=0):
        x1 += xoff
        x2 += xoff
        y1 += yoff
        y2 += yoff
        pygame.draw.line(self.screen,colour,(x1,y1),(x2,y1),thickness)
        pygame.draw.line(self.screen,colour,(x1,y1),(x1,y2),thickness)
        pygame.draw.line(self.screen,colour,(x2,y1),(x2,y2),thickness)
        pygame.draw.line(self.screen,colour,(x1,y2),(x2,y2),thickness)
    def mainloop(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.channels.timer_running=True
                
            
            self.convert_bot_camera_frame()  # Update cached frame if new one available
            self.convert_webcam_frame()      # Update webcam frame if new one available
            
            self.screen.blit(self.botcamframe, (0, 0))
            self.screen.blit(self.webcamframe,(640,0))
            pygame.draw.line(self.screen,(255,255,255),(0,1),(1280,1),width=4)
            pygame.draw.line(self.screen,(255,255,255),(0,480),(1280,480),width=4)
            pygame.draw.line(self.screen,(255,255,255),(640,0),(640,720),width=4)

            for x in range(5):
                if x == self.channels.phase:
                    self.screen.blit(self.phase_images_red[x],(496,480+(x*48)))
                else:
                    self.screen.blit(self.phase_images[x],(496,480+(x*48)))
            
            timer_value = self.channels.timer_value
            total_seconds = timer_value.total_seconds()
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            milliseconds = timer_value.microseconds // 1000
            self.screen.blit(self.render_text(f"{self.remzeroround(minutes,60)}:{self.remzeroround(seconds,60)}:{self.remzeroround(milliseconds,100)}"),(0,485))

            if self.channels.phase == 0:
                self.drawBoundingBox(100,100,300,300,xoff=640)
            # Update display
            pygame.display.flip()
            self.clock.tick(30)  # 30 FPS
        
        self.webcam_running = False
        self.webcam.release()
        pygame.quit()