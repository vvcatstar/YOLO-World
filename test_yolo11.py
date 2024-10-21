from ultralytics import YOLO
import cv2 

model = YOLO("/home/zyw/data/china_tower/CV_server/YOLO-World/runs/classify/train4/weights/best.pt")  # load a custom 
image = cv2.imread("/mnt/fillipo/yaowei/tower/annotation_data/crop_label/person/0_2.png")
results = model(image)  # predict on an image
results[0].plot()