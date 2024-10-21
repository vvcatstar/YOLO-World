from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")
# model.train(data='imagenet10', epochs=10, imgsz=128)
model.train(data='/mnt/fillipo/yaowei/tower/annotation_data/train_data/yolo_11/cls', epochs=40, imgsz=128)

