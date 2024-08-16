# Define the command trainYOLO to run train.py from YOLOv5 repository

trainF = [
    "python", "train.py",
    "--img", "640",
    "--batch", "16",
    "--epochs", "600",
    "--data", "C:/Users/.../coco-vehicle.yaml", #Path to TrafficFlowPopayan.yaml o coco-vehicle.yaml
    #Each yaml file are in the correspondet folder of the dataset
    "--weights", "yolov5l.pt", # You can use yolov5l_VC.pt for a retrainning
    "--freeze", "10", # Freeze backbone
    "--device", "0", # Use GPU in slot 1
    "--cache", "disk", # Store numpy file for each image to reduce ram consumption
]


destination_dir = 'C:/Users/.../yolov5-7.0'  # Change this path to the folder with YOLOv5 repository


import os
# Change the working directory to 'yolov5'
os.chdir(destination_dir)

import subprocess
import utils

display = utils.notebook_init()

subprocess.call(trainF)



