from ultralytics import YOLO
import cv2
model =YOLO("yolov8n.pt")

results = model(source=0,show=True,conf=0.4,save=False)