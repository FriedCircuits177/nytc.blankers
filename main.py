"""
nytc.blank

A collaborative effort by Rodin, Isaac, Bryan, Zexuan, Elliot for NYTC2026

Based on the UBTECH UGOT ecosystem in Mecanum mode.
"""

print(r"""
   ___   ___  ____  ___  ____  _        _    _   _ _  __  
  ( _ ) / _ \| ___|/ _ \| __ )| |      / \  | \ | | |/ /  
  / _ \| | | |___ \ (_) |  _ \| |     / _ \ |  \| | ' /   
 | (_) | |_| |___) \__, | |_) | |___ / ___ \| |\  | . \ _ 
  \___/ \___/|____/  /_/|____/|_____/_/   \_\_| \_|_|\_(_)
                                                          

 loading......

 importing modules...
""")

#import all necessary modules
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

#import definition files
import exceptions,robot,gui,process_manager

#notify user now!


if __name__ == "__main__":
    print(f"""
loading done! ({((datetime.now()-a).total_seconds())}s)

creating Manager instance and executing mainloops...""")

    manager = process_manager.Manager(ip="192.168.100.195")
    manager.mainloop()