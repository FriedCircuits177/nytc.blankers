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
# import pose_yolo
# importlib.reload(pose_yolo)
# from pose_yolo import run_pose_control_inline
from PIL import Image as Image2
from ultralytics import YOLO

from definitions import *

# COCO keypoint names for reference
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

class PoseDetector:
    """Handles pose detection and gesture classification for GUI."""
    
    def __init__(self, channels, model_path="yolov8n-pose.pt"):
        self.channels = channels
        self.model = YOLO(model_path)
        
        # Gesture classification thresholds
        self.up_margin_factor = 0.2
        self.down_margin_factor = 0.2
        self.min_conf = 0.3
        
        # Debouncing
        self.last_raw_command = "NONE"
        self.stable_command = "NONE"
        self.stable_count = 0
        self.debounce_frames = 5
        self.prev_stable_command = "NONE"
        
        # Visualization state
        self.current_keypoints = None
        self.current_frame = None
        
    def classify_pose(self, keypoints):
        """Classify pose into gesture commands."""
        
        def get(name):
            if name not in keypoints:
                return None
            x, y, c = keypoints[name]
            if c < self.min_conf:
                return None
            return np.array([x, y], dtype=np.float32)

        ls = get("left_shoulder")
        rs = get("right_shoulder")
        lw = get("left_wrist")
        rw = get("right_wrist")
        lh = get("left_hip")
        rh = get("right_hip")

        # Need at least shoulders + wrists
        if not all([ls is not None, rs is not None, lw is not None, rw is not None]):
            return "NONE"

        # Compute torso length (scale reference)
        torso_lengths = []
        if lh is not None:
            torso_lengths.append(np.linalg.norm(ls - lh))
        if rh is not None:
            torso_lengths.append(np.linalg.norm(rs - rh))
        if torso_lengths:
            torso = float(np.mean(torso_lengths))
        else:
            torso = float(np.linalg.norm(ls - rs))

        if torso < 1e-3:
            return "NONE"

        ls_y, rs_y = ls[1], rs[1]
        lw_y, rw_y = lw[1], rw[1]

        up_margin = self.up_margin_factor * torso
        down_margin = self.down_margin_factor * torso

        left_up = lw_y < ls_y - up_margin
        right_up = rw_y < rs_y - up_margin
        left_down = lw_y > ls_y + down_margin
        right_down = rw_y > rs_y + down_margin

        left_mid = not left_up and not left_down
        right_mid = not right_up and not right_down

        wrist_dx = abs(lw[0] - rw[0])

        # EXIT: deadzone + close together
        if left_mid and right_mid and wrist_dx < 0.4 * torso:
            return "EXIT"

        # PICKUP: deadzone + spread apart
        if left_mid and right_mid and wrist_dx > 2.5 * torso:
            return "PICKUP"

        # Movement poses
        if left_up and right_up:
            return "FORWARD"
        elif left_down and right_down:
            return "BACKWARD"
        elif left_up and not right_up:
            return "LEFT"
        elif right_up and not left_up:
            return "RIGHT"
        else:
            return "NONE"

    def draw_deadzone_band_cv2(self, frame, kps):
        """Draw deadzone band on opencv frame."""
        h, w, _ = frame.shape
        idx = {name: i for i, name in enumerate(COCO_KEYPOINTS)}

        def get_point(name):
            i = idx.get(name, None)
            if i is None:
                return None
            x, y, c = kps[i]
            if c < self.min_conf:
                return None
            return np.array([x, y], dtype=np.float32)

        ls = get_point("left_shoulder")
        rs = get_point("right_shoulder")
        lh = get_point("left_hip")
        rh = get_point("right_hip")

        if ls is None or rs is None:
            return

        torso_lengths = []
        if lh is not None:
            torso_lengths.append(np.linalg.norm(ls - lh))
        if rh is not None:
            torso_lengths.append(np.linalg.norm(rs - rh))
        if torso_lengths:
            torso = float(np.mean(torso_lengths))
        else:
            torso = float(np.linalg.norm(ls - rs))

        if torso < 1e-3:
            return

        ls_y, rs_y = ls[1], rs[1]
        up_margin = self.up_margin_factor * torso
        down_margin = self.down_margin_factor * torso

        top_y = int(min(ls_y - up_margin, rs_y - up_margin))
        bot_y = int(max(ls_y + down_margin, rs_y + down_margin))

        top_y = max(0, min(h - 1, top_y))
        bot_y = max(0, min(h - 1, bot_y))
        if bot_y <= top_y:
            return

        # Draw deadzone band
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, top_y), (w, bot_y), (255, 255, 0), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, dst=frame)
        
        cv2.line(frame, (0, top_y), (w, top_y), (0, 255, 255), 2)
        cv2.line(frame, (0, bot_y), (w, bot_y), (0, 255, 255), 2)
        cv2.putText(frame, "DEADZONE", (10, max(30, top_y - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def process_frame(self, frame):
        """Run pose detection on a frame and queue commands."""
        if frame is None or frame.size == 0:
            return

        # Run YOLO (synchronous, non-async)
        results = self.model(frame, verbose=False)
        raw_command = "NONE"

        if len(results) > 0:
            r = results[0]
            if r.keypoints is not None and len(r.keypoints) > 0:
                kps = r.keypoints[0].data[0].cpu().numpy()
                self.current_keypoints = kps

                keypoints_dict = {}
                for i, name in enumerate(COCO_KEYPOINTS):
                    x, y, c = kps[i]
                    keypoints_dict[name] = (float(x), float(y), float(c))

                raw_command = self.classify_pose(keypoints_dict)
                self.draw_deadzone_band_cv2(frame, kps)

                # Draw keypoints
                for x, y, c in kps:
                    if c > self.min_conf:
                        cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

        # Debounce
        if raw_command == self.last_raw_command:
            self.stable_count += 1
        else:
            self.last_raw_command = raw_command
            self.stable_count = 1

        if self.stable_count >= self.debounce_frames:
            self.stable_command = raw_command

        # Queue command to robot
        try:
            self.channels.pose_command_queue.put(self.stable_command, block=False)
        except:
            pass

        # Draw command on frame
        cv2.putText(frame, f"CMD: {self.stable_command}", (30, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw gesture guide text
        guide_text = [
            "FORWARD: both hands up",
            "BACKWARD: both hands down", 
            "LEFT: left hand up",
            "RIGHT: right hand up",
            "PICKUP: arms spread wide",
            "EXIT: hands together"
        ]
        for idx, text in enumerate(guide_text):
            cv2.putText(frame, text, (30, 100 + idx * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        self.current_frame = frame



class GUI():
    def __init__(self,channels,resolution=[1280,720]):
        
        self.channels = channels
        self.resolution = resolution
        
        #ensure window is centered
        os.environ['SDL_VIDEO_CENTERED'] = '1'

        pygame.init()
        pygame.mixer.init()

        self.screen = pygame.display.set_mode(resolution)
        self.clock = pygame.time.Clock()
        self.botcamframe = pygame.Surface(resolution)
        self.webcamframe = pygame.Surface((320, 240))  # Default webcam preview size
        self.pose_frame_display = pygame.Surface((320, 240))  # For pose detection visualization

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
            
        self.tone = pygame.mixer.Sound("assets/tone.wav")
        self.tone2 = pygame.mixer.Sound("assets/tone2.wav")
        self.tone3 = pygame.mixer.Sound("assets/tone3.wav")
        
        # Initialize pose detector
        self.pose_detector = PoseDetector(self.channels)
        self.pose_detection_thread = None
        
        print("\npygame GUI succesfully initialised.\n")

    def convert_bot_camera_frame(self):
        if not self.channels.camera_frame_queue.empty():
            frame = self.channels.camera_frame_queue.get()
            
            # Pygame's make_surface is very fast if the array is already 
            # (width, height, 3). 
            self.botcamframe = pygame.surfarray.make_surface(frame)
            #self.botcamframe = pygame.transform.scale(self.botcamframe,(640,480))
    
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
    
    def update_pose_detection(self):
        """Process pose detection from webcam frame and update display."""
        # Get the latest webcam frame
        try:
            frame = self.channels.webcam_frame_queue.get(block=False)
            # Process frame for pose detection
            self.pose_detector.process_frame(frame.copy())
        except:
            pass
        
        # Convert processed frame to pygame surface if available
        if self.pose_detector.current_frame is not None:
            frame_rgb = cv2.cvtColor(self.pose_detector.current_frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (320, 240))  # Resize to fit display area
            frame_transposed = np.transpose(frame_rgb, (1, 0, 2))
            self.pose_frame_display = pygame.surfarray.make_surface(frame_transposed)
    
    def mainloop(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.channels.timer_running:
                        # Stop the timer and robot
                        self.channels.timer_running = False
                        self.channels.phase = 0
                        self.channels.initialise()
                        
                    else:
                        # Start the timer
                        self.channels.timer_running = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                        if self.channels.timer_running:
                        # Stop the timer and robot
                            self.channels.timer_running = False
                            self.channels.phase = 0
                            self.channels.initialise()
                            self.robot.subinit()
                    else:
                        # Start the timer
                        self.channels.timer_running = True

                
            
            self.convert_bot_camera_frame()  # Update cached frame if new one available
            self.convert_webcam_frame()      # Update webcam frame if new one available
            
            # Update pose detection if phase 3 (pose drive)
            if self.channels.phase == 3:
                self.update_pose_detection()
            
            self.screen.fill((0, 0, 0))  # Clear screen before drawing new frame
            self.screen.blit(self.botcamframe, (0, 0))
            
            # Show pose detection visualization during phase 3, otherwise regular webcam
            if self.channels.phase == 3:
                self.pose_frame_display = pygame.transform.scale(self.pose_frame_display,(640,480))
                self.screen.blit(self.pose_frame_display, (640, 0))
                # Draw "POSE DETECTION" label
                pose_label = self.render_text("POSE DETECTION", size=18, colour=(0, 255, 0))
                self.screen.blit(pose_label, (645, 5))
            else:
                self.screen.blit(self.webcamframe, (640, 0))
            
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

            if not self.channels.sound_queue.empty() and not pygame.mixer.get_busy():
                a = self.channels.sound_queue.get(block=False)
                if a == 0:
                    self.tone.play()
                elif a == 1:
                    self.tone2.play()
                elif a == 2:
                    self.tone3.play()
            self.clock.tick(30)  # 30 FPS
        
        self.webcam.release()
        pygame.quit()