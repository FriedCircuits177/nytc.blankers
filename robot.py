"""
robot.py

Contains the class definition for the Robot object.
"""
import pygame
import time
import importlib
import cv2
import numpy as np
from datetime import datetime
from ugot import ugot
# import pose_yolo
# importlib.reload(pose_yolo)
# from pose_yolo import run_pose_control_inline
from IPython.display import display, clear_output, Image
from PIL import Image as Image2
# from pose_yolo import run_pose_control_inline
import threading
import definitions

class Robot():
    def __init__(self,channels,ip='192.168.88.1',config={}):
        self.channels = channels
        self.robot = ugot.UGOT()
        self.config = config

        #try-except block to hopefully catch any connection errors
        try:
            print("attempting to connect to robot at:")
            self.robot.initialize(ip)
        except:
            raise definitions.InvalidUgotIP(f"Failed to connect to robot at IP {ip}.\nPlease check the WiFi connection and IP shown on the bot.")
        
        print("\nrobot connected succesfully!")
        self.robot.load_models(
            ["color_recognition", "word_recognition", "line_recognition", "apriltag_qrcode"]
        )
        self.robot.open_camera()
        self.robot.set_track_recognition_line(0)
        self.robot.screen_display_background(0)

        #multithread the camera updates always, always, always
        self.camera_thread = threading.Thread(target=self.update_camera_frame)
        self.camera_thread.start()

        self.robot.mechanical_joint_control(0,30,30,1000)
        self.robot.screen_display_background(0)
        self.robot.mechanical_clamp_close()

    def subinit(self):
        self.robot.mecanum_stop()
        self.robot.set_track_recognition_line(0)
        self.robot.screen_display_background(0)
        self.robot.mechanical_joint_control(0,30,30,1000)
        
        self.robot.screen_display_background(0)

    def line_follow(self, mult=0.25, speed=35):
        """Follow the detected line by turning proportionally to the line offset."""

        offset, line_type, x, y = self.robot.get_single_track_total_info()
        rotation_speed = int(offset * mult)
        self.robot.mecanum_move_xyz(x_speed=0, y_speed=speed, z_speed=rotation_speed)
        return line_type, x, y
    def update_camera_frame(self):
        while True:
            frame = self.robot.read_camera_data()
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
    
    def colourdetect(self):
        """Perform colour detection on the latest HSV frame.
        Returns: detection_result dict or None if no colour detected."""
        # Get the latest HSV frame (non-blocking)
        try:
            frame = self.channels.hsv_camera_frame_queue.get(block=False)
        except:
            return None
        
        if frame is None or frame.size == 0:
            return None
        
        # Find the most dominant color using histogram on H channel (efficient)
        h_channel = frame[:, :, 0]
        hist = cv2.calcHist([h_channel], [0], None, [180], [0, 180])
        most_common_hue = int(np.argmax(hist))
        
        # Create mask for colors near this hue
        lower = np.array([max(0, most_common_hue - 15), 30, 30])
        upper = np.array([min(180, most_common_hue + 15), 255, 255])
        mask = cv2.inRange(frame, lower, upper)
        
        # Remove noise with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (most prominent)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h_box = cv2.boundingRect(largest_contour)
            
            # Calculate mean color within bounding box
            roi = frame[y:y+h_box, x:x+w]
            mean_hsv = cv2.mean(roi)[:3]
            
            detection_result = {
                'hue': int(mean_hsv[0]),
                'sat': int(mean_hsv[1]),
                'val': int(mean_hsv[2]),
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h_box),
                'area': int(cv2.contourArea(largest_contour))
            }
            
            if not self.channels.color_detection_queue.full():
                self.channels.color_detection_queue.put(detection_result)
            
            return detection_result
        
        return None
    
    def apriltagcentre(self, distance=0.250, nudge = 4, gap=20, fwd_spd=15, strafe_spd=10, timeout=10):
        """
        Drive toward a detected AprilTag, keeping it centered in the camera frame.

        Parameters:
            distance  (float): Stop when the tag is within this many meters (default 0.15 m).
            gap       (int):   Pixel tolerance around center (320 px) before strafing (default 20 px).
            fwd_spd   (int):   Forward drive speed percentage (default 10 cm/s).
            strafe_spd(int):   Left/right correction speed percentage (default 10 cm/s).
            timeout   (int):   Maximum seconds to search for tag (default 10s).
        """
        AP_distance = 100000
        self.AP_info = self.robot.get_apriltag_total_info()
        # Refresh tag data every iteration for responsive corrections.
        while True:
            self.AP_info = self.robot.get_apriltag_total_info()
            try:
                AP_x = self.AP_info[0][1]
                AP_distance = self.AP_info[0][6]
                tag_found = True
            except (IndexError, TypeError):
                if AP_distance < distance:
                    break
                print("no tag detected")
                time.sleep(0.1)
                continue
            print(str(AP_distance))
            if AP_x < 320 - gap:
                # Tag is to the LEFT of center — strafe left to re-align.
                # mecanum_move_xyz(x, y, z): x=strafe, y=forward, z=rotation
                self.robot.mecanum_move_xyz(-strafe_spd, strafe_spd, 0)
            elif AP_x > 320 + gap:
                # Tag is to the RIGHT of center — strafe right to re-align.
                self.robot.mecanum_move_xyz(strafe_spd, strafe_spd, 0)
            elif AP_distance > distance:
                # Tag is centered but still too far — drive straight forward.
                self.robot.mecanum_move_xyz(0, fwd_spd, 0)
            else:
                # Tag is centered AND within target distance — stop and exit.
                print("It's too close, let's stop.")
                self.robot.mecanum_move_speed_times(0,7,nudge,1)
                #time.sleep(1)
                self.pick_up(self.AP_info)
                break

    def pick_up(self,AP_info):
        """Pick up an object using the arm based on the AprilTag position."""

        while True:
            try:
                AP_x = AP_info[0][1]
                AP_distance = AP_info[0][6]
                break
            except:
                raise IndexError("DUMMY DIDNT GIVE ME MY INFO")

        # Move arm to a neutral ready position and open the gripper.
        # joint_control(j1, j2, j3, duration_ms): j2=30, j3=30 tilts arm slightly forward.
        #self.robot.mechanical_joint_control(0, 30, 30, 500)
        self.robot.mechanical_clamp_release() # Open gripper before extending arm
        time.sleep(0.5) # Wait for gripper to fully open

        # Calculate arm joint angles based on the tag's camera position.
        # joint1 (base): convert pixel offset from center to degrees.
        #   Negative factor corrects for the camera being mirrored horizontally.
        joint1 = int((AP_x - 320) * -1 / 10)

        # joint3 (furthest): convert distance (m) to an extension angle.
        # The -80 offset accounts for the arm's resting angle calibration.
        joint3 = int(AP_distance * 100 - 80)

        # Move arm to the computed pick-up pose.
        self.robot.mechanical_joint_control(joint1, -10, joint3, 500)
        print(f"Joint1 value is: {joint1}, Joint3 value is: {joint3}.")
        time.sleep(1) # Wait for arm to reach the target pose

        # Grasp the object and lift the arm back to the carry position.
        self.robot.mechanical_clamp_close()
        time.sleep(0.2)  # Wait for gripper to fully close before lifting
        self.robot.mechanical_joint_control(0, 30, 30, 500)  # Return arm to neutral carry pose
    
    def phase1(self):
        print("[robot] PHASE 1")
        
        while True:
            if not self.channels.timer_running:
                self.robot.mecanum_stop()
                return
                
            #search for green
            detection = self.robot.get_color_total_info()[0]
            
            if detection:
                # Red: hue < 20 or hue > 160
                if detection == "Red":
                    self.robot.screen_display_background(3)
                    print("red")
                    
                # Green: hue 40-80
                elif detection == "Green":
                    self.robot.screen_display_background(6)
                    print("green")
                    break
            
            time.sleep(0.25)
        self.robot.mechanical_joint_control(0,0,0,500)
        print("green detected moving on")
        #time.sleep(1)

        
        self.apriltagcentre()
        time.sleep(0.5)

    def phase2(self):
        print("[robot] PHASE 2")
        
        #REMEMBER TO TUNE THESE
        BRANCH_ANGLE = 35
        DUMB_SECONDS = 1
        #DONE

        
        text = "I_DONT_KNOWWWWWW"
        turned = "not yet"


        print(f"Phase 2 following line")
        self.robot.mecanum_move_speed_times(0,70,80,1)
        # while True:
        #     if not self.channels.timer_running:
        #         self.robot.mecanum_stop()
        #         return

        #     offset, line_type, x, y = self.robot.get_single_track_total_info()           

        #     print("Info:", {offset, line_type, x, y})
        #     self.line_follow(mult=0.25, speed=20)

        #     if line_type == 2:
        #         print("INTERSECTION!!!!!!!!!!!!!")
        #         break

        while True:
            if not self.channels.timer_running:
                self.robot.mecanum_stop()
                return
                
            self.robot.mecanum_stop()
            text = self.robot.get_words_result()
            print("Detecting Sign, text is:", text)
                        
            if text == "LEFT":
                print("LEFT FOUND")
                turned = "LEFT"
                self.robot.mecanum_move_speed_times(0,30,5,1)
                self.robot.mecanum_turn_speed_times(turn=2, speed=BRANCH_ANGLE*2, times= BRANCH_ANGLE, unit = 2)
                #self.robot.mecanum_move_speed_times(0, 20, DUMB_SECONDS, 0)
                break
            elif text == "RIGHT":
                print("RIGHT FOUND")
                turned = "RIGHT"
                self.robot.mecanum_move_speed_times(0,30,5,1)
                self.robot.mecanum_turn_speed_times(turn=3, speed=BRANCH_ANGLE*2, times= BRANCH_ANGLE, unit=2)
                #self.robot.mecanum_move_speed_times(0, 20, DUMB_SECONDS, 0)            
                break    
            else:
                self.robot.mecanum_move_speed_times(0,10,5,1)
                time.sleep(0.2)

        loop_count = 0
        line_type = 1
        while True:
            if not self.channels.timer_running:
                self.robot.mecanum_stop()
                return
                
            print(f"Phase 2.5 following line", line_type)
            line_type, _, _ = self.line_follow(mult=0.25, speed=30)

            if line_type == 0:
                print("NO LINE!!!!!!!!!!!")
                self.robot.mecanum_move_speed_times(1, 10, 1, 0)

            elif line_type == 2 and loop_count > 30:
                print("INTERSECTION!!!!!!!!!!!!!")
                if turned == "LEFT":
                    print("TURNING LEFT")
                    self.robot.mecanum_move_speed_times(0,30,15,1)
                    self.robot.mecanum_turn_speed_times(turn=2, speed=BRANCH_ANGLE*2, times= BRANCH_ANGLE, unit = 2)
                    #self.robot.mecanum_move_speed_times(0, 20, DUMB_SECONDS, 0)
                    break
                elif turned == "RIGHT":
                    print("TURNING RIGHT")
                    self.robot.mecanum_move_speed_times(0,30,15,1)
                    self.robot.mecanum_turn_speed_times(turn=3, speed=BRANCH_ANGLE*2, times= BRANCH_ANGLE, unit=2)
                    #self.robot.mecanum_move_speed_times(0, 20, DUMB_SECONDS, 0)            
                    break
            loop_count += 1

        loop_count = 0
        trace = True

        while True:
            if not self.channels.timer_running:
                self.robot.mecanum_stop()
                return
                
            print(f"Phase 2.8 following line", line_type)
            line_type, _, _ = self.line_follow(mult=0.25, speed=20)

            if line_type == 0:
                if trace == True:
                    print("ending phase 2")
                    self.robot.mecanum_stop()
                    break        
            #    elif trace == False:
            #        while line_type == 0:
            #            print("finding line")         
            #            self.robot.mecanum_move_speed_times(0, 10, 5, 1)
            #            offset, line_type, x, y = self.robot.get_single_track_total_info()           
            #        trace = True





    def posedrive3(self):
        """Phase 3: Pose-based gesture control for the robot.
        
        Gestures:
        - FORWARD: both hands raised -> drive forward
        - BACKWARD: both hands lowered -> drive backward
        - LEFT: left hand raised -> turn left
        - RIGHT: right hand raised -> turn right
        - PICKUP: arms spread wide at shoulder height -> execute pickup sequence
        - EXIT: hands close together at shoulder height -> stop and exit
        - NONE: no valid gesture -> stop
        
        During startup delay period (configurable), no commands register.
        """
        print("[robot] PHASE 3 POSE DRIVE")
        self.robot.mecanum_stop()
        
        forward_speed = 30
        backward_speed = 30
        turn_speed = 45
        last_command = "NONE"
        command = "NONE"
        commands_received = 0
        
        #delay 1.5 seconds for coordination
        time.sleep(1.5)
        
#        try:



        while True:
            # Exit if timer stops
            if not self.channels.timer_running:
                self.robot.mecanum_stop()
                break
            
            command = "NONE"

            if not self.channels.pose_command_queue.empty():
                command = self.channels.pose_command_queue.get(block=False)
            
            print(f"pose driving,", {command}, {last_command})

            
            # if command == last_command and command == "NONE":
            #     self.robot.mecanum_stop()

            if command != last_command:
                commands_received += 1
                
                if command == "FORWARD":
                    self.robot.mecanum_move_speed_times(0, forward_speed, times=5, unit=1)
                elif command == "BACKWARD":
                    self.robot.mecanum_move_speed_times(1, backward_speed, times=5, unit=1)
                elif command == "LEFT":
                    self.robot.mecanum_turn_speed_times(turn=2, speed=turn_speed, times=30, unit=2)
                elif command == "RIGHT":
                    self.robot.mecanum_turn_speed_times(turn=3, speed=turn_speed, times=30, unit=2)
                elif command == "EXIT" and commands_received > 3:
                    self.robot.mecanum_stop()
                    print("EXIT gesture detected. Exiting pose drive.")
                    break

                last_command = command

            time.sleep(0.02)
        
#        except Exception as e:
#            print(f"Error in posedrive3: {e}")
#            self.robot.mecanum_stop()
#            return
    
    def driveanddrop(self,distance=15,speed=60):
        self.robot.mecanum_stop()
        self.robot.mecanum_move_speed_times(0,speed,distance,1)
        self.robot.mechanical_joint_control(0,-20,-70,500)
        self.robot.mechanical_clamp_release()
        exit()
        
    # def phase3(self):
    #     print("[robot] PHASE 3, FACE RECOGNITION")
    #     #REMEMBER MEEEEE
    #     TARGET_NAME = "Ryan"
    #     gap = 10        
    #     #DONE
    #     target_name = TARGET_NAME
    #     turn_spd = 30
    #     strafe_spd = 25
    #     fwd_spd = 5 #speed
    #     height = 40 #distance
    #     adjust_turn = 15
    #     face_name = None

    #     X_TOLERANCE_ON_FIRST = 100

    #     print("p3p1")

    #     while True:
    #         if not self.channels.timer_running:
    #             self.robot.mecanum_stop()
    #             return
                
            
    #         self.robot.mecanum_turn_speed(turn=3, speed=turn_spd)

    #         name = self.robot.get_words_result()

    #         print(f"{name}, {face_name}")
    #         # Check for any recognized faces in the frame
    #         faces = self.robot.get_face_recognition_total_info()
    #         if faces:
    #             face_name = faces[0][0]  # We need to calibrate all the face first
    #             if face_name != TARGET_NAME:
    #                 while not (faces[0][1] < (640/2)+X_TOLERANCE_ON_FIRST and faces[0][1] > (640/2)-X_TOLERANCE_ON_FIRST):
    #                     self.robot.mecanum_turn_speed_(turn=3, speed=turn_spd)
    #                 break
    #             elif face_name == TARGET_NAME:
    #                 self.driveanddrop()

    #     while True:     
    #         faces = self.robot.get_face_recognition_total_info()
    #         name = self.robot.get_words_result()
    #         if name == target_name or face_name == target_name:
    #             self.robot.mecanum_stop()
    #             print(f"Saw {target_name}!")

    #             # Small corrective turn to center the robot on the target
    #             self.robot.mecanum_turn_speed_times(turn=3, speed=20, times=adjust_turn, unit=2)
    #             break  

    #     print("p3p2")

    #     face_counter = 0

    #     while True:
    #         if not self.channels.timer_running:
    #             self.robot.mecanum_stop()
    #             return
                
    #         name = self.robot.get_words_result()
    #         faces = self.robot.get_face_recognition_total_info()

    #         if not faces:
    #             # Lost the face; inch forward slowly to try to find it again
    #             self.robot.mecanum_translate_speed(angle=0, speed=fwd_spd)
    #             face_counter += 1
    #             if face_counter > 20:
    #                 break
    #         else:
    #             c_x = faces[0][1]  # Horizontal center of the face in the frame (0–640 px)
    #             h = faces[0][3]  # Height of the face bounding box (proxy for distance)
    #             if h < height:
    #                 if c_x < 320 - gap:
    #                     # Face is too far LEFT — strafe left while moving forward
    #                     self.robot.mecanum_move_xyz(
    #                         x_speed=-strafe_spd, y_speed=fwd_spd, z_speed=0
    #                     )
    #                 elif c_x > 320 + gap:
    #                     # Face is too far RIGHT — strafe right while moving forward
    #                     self.robot.mecanum_move_xyz(x_speed=strafe_spd, y_speed=fwd_spd, z_speed=0)
    #                 else:
    #                     # Face is centered but still small (far away) — move straight forward
    #                     self.robot.mecanum_move_xyz(x_speed=0, y_speed=fwd_spd, z_speed=0)
    #             else:
    #                # Face is centered AND large enough — we've arrived!
    #                self.robot.mecanum_stop()
    #                print(f"Reached {target_name}!")
    #                break  # Done
    #         print(f"alignment:", {c_x}, {h})

    #     clear_output(wait=True)
    #     self.robot.mecanum_stop()
    #     self.robot.mechanical_clamp_release()
    def phase3(self):
        print("[robot] PHASE 3, FACE RECOGNITION")

        #REMEMBER MEEEEE
        TARGET_NAME = "Ryan"
        # TARGET_NAME = "Ananya"
        # TARGET_NAME = "Coley"
        # TARGET_NAME = "Arjun"
        gap = 10        
        #DONE
        target_name = TARGET_NAME
        turn_spd = 30
        strafe_spd = 25
        fwd_spd = 5 #speed
        height = 40 #distance
        adjust_turn = 15
        face_name = None

        print("p3p1")

        while True:
            if not self.channels.timer_running:
                self.robot.mecanum_stop()
                return

            name = self.robot.get_words_result()

            #print(f"{name}, {face_name}")
            # Check for any recognized faces in the frame
            faces = self.robot.get_face_recognition_total_info()
            if faces:
                face_name = faces[0][0]  # We need to calibrate all the face first


            if name == target_name or face_name == target_name:
                self.robot.mecanum_stop()
                print(f"Saw {target_name}!")

                # Small corrective turn to center the robot on the target
                self.robot.mecanum_turn_speed_times(turn=3, speed=10, times=10, unit=2)
                break

            if name == "Ryan" or name == "Coley" or name == "Arjun" or name == "Ananya":
                self.robot.mecanum_turn_speed_times(turn=3, speed=80, times=50, unit=2)

            self.robot.mecanum_turn_speed_times(turn=3, speed=20, times=10, unit=2)



        self.driveanddrop()
 

    
    def mainloop(self):
        while True:
            # If timer stops, always reset to phase 0 and stop robot
            if not self.channels.timer_running:
                self.channels.phase = 0
                self.robot.mecanum_stop()
            
            # Phase 0: Standby
            if self.channels.phase == 0:
                self.robot.mecanum_stop()
                
                if self.channels.timer_running:
                    self.channels.phase = self.channels.start_phase
            
            # Phase 1
            elif self.channels.phase == 1:
                self.channels.timer_value = datetime.fromtimestamp(0)-datetime.fromtimestamp(0)
                self.channels.sound_queue.put(1)
                self.phase1()
                self.channels.phase = 2
            
            # Phase 2
            elif self.channels.phase == 2:
                self.channels.sound_queue.put(2)
                self.phase2()
                self.channels.phase = 3
            
            # Phase 3
            elif self.channels.phase == 3:
                self.channels.sound_queue.put(2)
                self.channels.sound_queue.put(2) #Intentional to play twice
                self.posedrive3()
                self.channels.phase = 4
            
            # Phase 4
            elif self.channels.phase == 4:
                self.channels.sound_queue.put(1)
                self.phase3()
                self.channels.phase = 0
                self.channels.sound_queue.put(0)