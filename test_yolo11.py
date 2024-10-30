from ultralytics import YOLO
import cv2 

# test cls
# model = YOLO("/home/zyw/data/china_tower/CV_server/YOLO-World/runs/classify/train4/weights/best.pt")  # load a custom 
# image = cv2.imread("/mnt/fillipo/yaowei/tower/annotation_data/crop_label/person/0_2.png")
# results = model(image)  # predict on an image

# test detection 
# Load a model
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# # Train the model
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)