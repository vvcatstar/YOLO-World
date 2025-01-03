import requests
import os 
import json
import yaml 
import supervision as sv
import numpy as np 
import pandas as pd 
import cv2
import pickle
import time 
from supervision.draw.color import ColorPalette
from flask import Flask, request, jsonify
from ultralytics import YOLO
import supervision as sv
from supervision import box_non_max_suppression
from IPython import embed

CUSTOM_COLOR_MAP = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
]
class PostProcess:
    def __init__(self, config):
        self.onnx_model = config['onnx']
        if self.onnx_model:
            self.smoke_cls = YOLO(config['post_config']['smoke_cls_onnx'], task='classify')
            self.fire_env_model = YOLO(config['post_config']['fire_det_onnx'], task='detect')
            self.person_det = YOLO(config['post_config']['person_det_onnx'], task='detect')
            # self.person_det.set_classes(["person"])
        else:
            self.smoke_cls = YOLO(config['post_config']['smoke_cls'])
            self.fire_env_model = YOLO(config['post_config']['fire_det'])
            self.person_det = YOLO(config['post_config']['person_det'])
            self.person_det.set_classes(["person"])
        self.gd_server = config['post_config']['gd_server']
        self.config = config 
        self.post_func = {
            'black_smoke': self.post_smoke,
            'fishing': self.post_fishing,
            'boat': self.post_boat,
            'forest': self.post_fire,
            'factory': self.post_fire,
            'farmland': self.post_fire,
            'fire': self.post_fire,
            'traffic': self.post_traffic,
        }
        self.label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        self.box_annotator = sv.BoundingBoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        return 
        
    def detection_filter(self, outputs, prompt):
        filter_classes = prompt.strip('.').split('.')
        filtered_outputs = {
            'image_path': outputs['image_path'],
            'xyxy': [],
            'confidence': [],
            'class_name': []
        }
        for i, class_name in enumerate(outputs['class_name']):
            if class_name in filter_classes:
                filtered_outputs['xyxy'].append(outputs['xyxy'][i])
                filtered_outputs['confidence'].append(outputs['confidence'][i])
                filtered_outputs['class_name'].append(class_name)
        return filtered_outputs
    
    def post_boat(self, results):
        image = cv2.imread(results['image_path'])
        input_boxes = results['xyxy']
        class_names = results['class_name']
        person_details = []
        for index in range(len(input_boxes)):
            if class_names[index] in ['small rowboat', 'raft', 'rowboat']:
                bbox = input_boxes[index]
                x_min, y_min, x_max, y_max = map(int, bbox)
                width = x_max - x_min
                height = y_max - y_min
                new_x_min = max(x_min - width , 0)
                new_y_min = max(y_min - height , 0)
                new_x_max = min(x_max + width , image.shape[1])
                new_y_max = min(y_max + height , image.shape[0])
                cropped_image = image[new_y_min:new_y_max, new_x_min:new_x_max]
                person_result = self.person_det(cropped_image, conf=0.1, iou=0.45)[0]
                for j in range(len(person_result.boxes)):
                    p_x1 = float(person_result.boxes.xyxy[j][0].detach().cpu().numpy() + new_x_min)
                    p_y1 = float(person_result.boxes.xyxy[j][1].detach().cpu().numpy() + new_y_min)
                    p_x2 = float(person_result.boxes.xyxy[j][2].detach().cpu().numpy() + new_x_min)
                    p_y2 = float(person_result.boxes.xyxy[j][3].detach().cpu().numpy() + new_y_min)
                    conf = float(person_result.boxes.conf[j].detach().cpu().numpy())
                    person_details.append([p_x1, p_y1, p_x2, p_y2, conf, 'person'])
        for person in person_details:
            results['xyxy'].append(person[:4])
            results['class_name'].append(person[5])
            results['confidence'].append(person[4])
        return results 
    
    def post_smoke(self, results):
        # use finetune cls model to fix FP
        image = cv2.imread(results['image_path'])
        input_boxes = results['xyxy']
        class_names = results['class_name']
        confidences = results['confidence']
        for index in range(len(input_boxes)):
            if class_names[index] in ['billowing black smoke', 'smoke']:
                bbox = input_boxes[index]
                x_min, y_min, x_max, y_max = map(int, bbox)
                cropped_image = image[y_min:y_max, x_min:x_max]
                # if self.onnx_model:
                #     cropped_image = cv2.resize(cropped_image, (128, 128))
                cls_result = self.smoke_cls(cropped_image)[0]
                cls_name = cls_result.names[cls_result.probs.top1]
                print(f'{class_names[index]}->{cls_name}')
                class_names[index] = cls_name
                confidences[index] = float(cls_result.probs.top1conf.detach().cpu())
        # use finetune detection model to fix FN
        det_results = self.fire_env_model(results['image_path'], conf=0.1, iou=0.5, agnostic_nms=True)[0]
        xyxy = det_results.boxes.xyxy.detach().cpu().numpy().tolist()
        conf = det_results.boxes.conf.detach().cpu().numpy().tolist()
        cls_id = det_results.boxes.cls.detach().cpu().numpy().tolist()
        class_name = [det_results.names[id] for id in cls_id]
        for index, name in enumerate(class_name):
            if name in ['billowing smoke', 'white smoke', 'fire']:
                results['xyxy'].append(xyxy[index])
                if name == 'billowing smoke':
                    results['class_name'].append('billowing')
                else:
                    results['class_name'].append(name)
                results['confidence'].append(conf[index])
        return results
    
    def post_fire(self, results):
        # image = cv2.imread(results['image_path'])
        det_results = self.fire_env_model(results['image_path'], conf=0.1, iou=0.5, agnostic_nms=True)[0]
        xyxy = det_results.boxes.xyxy.detach().cpu().numpy().tolist()
        conf = det_results.boxes.conf.detach().cpu().numpy().tolist()
        cls_id = det_results.boxes.cls.detach().cpu().numpy().tolist()
        class_name = [det_results.names[id] for id in cls_id]
        for index in range(len(xyxy)):
            results['xyxy'].append(xyxy[index])
            results['class_name'].append(class_name[index])
            results['confidence'].append(conf[index])
        text = 'forest.field.farmland.'
        gd_inputs = {
            'image_path': results['image_path'],
            'text_prompt': text,
        }
        # try:
        response = requests.post(self.gd_server, json=gd_inputs, headers={
            "accept": "application/json",
            "Content-Type": "application/json",
        })
        data = response.json()
        for index in range(len(data['class_name'])):
            results['xyxy'].append(data['xyxy'][index])
            results['class_name'].append(data['class_name'][index])
            results['confidence'].append(data['confidence'][index])
        task_prompt = self.config['tasks']['forest']['prompt']
        filter_results = self.detection_filter(results, task_prompt)
        return filter_results    

    def post_fishing(self, results):
        return results
    
    def post_traffic(self, results):
        return results
    
    def nms_func(self, results):
        input_boxes = results['xyxy']
        class_names = results['class_name']
        confidences = results['confidence']
        class_ids = np.array(list(range(len(class_names))))
        if len(class_names):
            detections = sv.Detections(
                xyxy=np.array(input_boxes),  # (n, 4)
                class_id=class_ids,
                confidence=np.array(confidences),
            )
            detections = detections.with_nms(threshold=0.5, class_agnostic=True)
            nms_class_ids = detections.class_id
            nms_class_names = []
            for id in nms_class_ids:
                nms_class_names.append(class_names[id])
            class_names = [class_names[i] for i in class_ids]
            results['xyxy'] = detections.xyxy.tolist()
            results['class_name'] = nms_class_names
            results['confidence'] = detections.confidence.tolist()
        return results
    
    def show_func(self, results):
        annotated_frame = cv2.imread(results['image_path'])
        unique_classes = list(set(results['class_name']))
        class_id_map = {name: idx for idx, name in enumerate(unique_classes)}
        class_ids = np.array([class_id_map[class_name] for class_name in results['class_name']])

        # 将其他数据转换为 np.array
        input_boxes = np.array(results['xyxy'])
        if input_boxes.shape[0]:
            confidences = np.array(results['confidence'])
            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence
                in zip(results['class_name'], confidences)
            ]
            detections = sv.Detections(
                xyxy=input_boxes,
                class_id=class_ids,
                confidence=confidences,
            )
            annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels)
        return annotated_frame
    
    def post_task(self, task, results):
        task_prompt = self.config['tasks'][task]['prompt']
        filter_results = self.detection_filter(results, task_prompt)
        post_func = self.post_func[task]
        if post_func:
            post_results = post_func(filter_results)
        nms_results = self.nms_func(post_results)
        annotate_image = self.show_func(nms_results)
        return post_results, annotate_image

# with open('./config.yaml', 'r') as f:
#     configs = yaml.load(f, Loader=yaml.Loader)
# post_processer = PostProcess(configs)
# app = Flask(__name__)
# @app.route('/yolo_detection', methods=['POST'])
# def post_process():
#     start_time = time.time()
#     data = request.get_json()
#     image_path = data['image_path']
#     task = data['task']
#     output_root = data.get('output_root', output_root)
#     output_file = post_processer.post_task(task, det_result)


# if __name__ == '__main__':
#     eval_result = '/mnt/fillipo/yaowei/tower/test_output/1028_merge/eval.xlsx'
#     with open('./config.yaml', 'r') as f:
#         configs = yaml.load(f, Loader=yaml.Loader)
#     post_processer = PostProcess(configs)
#     for task in ['black_smoke', 'boat']:
#         eval_data = pd.ExcelFile(eval_result).parse('boat')
#         for index in range(len(eval_data)):
#             data = eval_data.iloc[index]
#             det_file = data['det_result']
#             with open(det_file, 'r') as f:
#                 det_result = json.load(f)
#             post_processer.post_task(task, det_result)
        
        
        
        
    
    
    