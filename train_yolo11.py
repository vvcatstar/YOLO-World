from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")
# model = YOLO("/home/zyw/data/china_tower/CV_server/YOLO-World/runs/classify/train14/weights/last.pt")


# model.train(data='imagenet10', epochs=10, imgsz=128)
model.train(data='/mnt/fillipo/yaowei/tower/annotation_data/train_data/yolo_11/fire/', epochs=150, imgsz=128)

