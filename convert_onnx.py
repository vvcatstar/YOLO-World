from ultralytics import YOLO
import os 
import cv2 

# smoke_cls_model_checkpoints = '/home/zyw/data/china_tower/CV_server/weights/torch/smoke_cls.pt'
# smoke_model = YOLO(smoke_cls_model_checkpoints, task='classify')
# smoke_model.export(format='onnx', dynamic=True)

# smoke_model = YOLO('/home/zyw/data/china_tower/CV_server/weights/torch/smoke_cls.onnx', task='classify')
# image = cv2.imread('/mnt/fillipo/yaowei/tower/annotation_data/crop_label/billowing smoke/817_6.png')
# print(image.shape)

# detection_checkpoints = '/home/zyw/data/china_tower/CV_server/ultralytics/runs/detect/train6/weights/best.pt'
# det_model = YOLO(detection_checkpoints)
# det_model.export(format='onnx')
# det_onnx_model = YOLO('/home/zyw/data/china_tower/CV_server/weights/onnx/detection.onnx', task='detect')
# results = det_onnx_model('/mnt/fillipo/yaowei/tower/复杂任务精确率提升基准及测试数据/测试数据/散煤冒黑烟/测试正样本-50/1722832255561.jpg')
# print(results)

# person_det_checpoints = '/home/zyw/data/china_tower/CV_server/yolov8x-worldv2.pt'
# person_model = YOLO(person_det_checpoints)
# person_model.set_classes(["person"])
# person_model.export(format='onnx')

