"""Module providing a functions to process task and process image in tower situation."""
import os
import glob
import json
import pickle
import yaml
import os.path as osp
import cv2
import torch
import numpy as np
import supervision as sv
import sys 
from PIL import Image
from ultralytics import YOLO
import supervision as sv
from supervision.draw.color import ColorPalette
import tqdm
import random
import time
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
from flask import Flask, request, jsonify
from IPython import embed 
from post_process import PostProcess
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
class YOLO_WORLD:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.configs = yaml.load(f, Loader=yaml.Loader)
        with open('/home/zyw/data/china_tower/CV_server/config.yaml','r') as f:
            self.post_config = yaml.load(f, Loader=yaml.Loader)
        self.config_file = self.configs['config_file']
        self.checkpoint = self.configs['checkpoint']
        self.post_processer = PostProcess(self.post_config)
        self.cfg = Config.fromfile(self.config_file)
        self.cfg.work_dir = osp.join('./work_dirs')
        self.cfg.load_from = self.checkpoint
        # self.model and self.test_pipeline use to process image
        self.model = init_detector(self.cfg, checkpoint=self.checkpoint, device='cuda:0')
        # self.smoke_model = YOLO("/home/zyw/data/china_tower/CV_server/YOLO-World/runs/classify/smoke/weights/best.pt")
        # self.boat_model = YOLO("/home/zyw/data/china_tower/CV_server/YOLO-World/runs/classify/train18/weights/best.pt")
        self.test_pipeline_cfg = get_test_pipeline_cfg(cfg=self.cfg)
        self.test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
        self.test_pipeline = Compose(self.test_pipeline_cfg)
        # use show bbox result for debug
        self.label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        self.box_annotator = sv.BoundingBoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        
    def show_bbox(self, image_path, detections, labels, output_image):
        img = cv2.imread(image_path)
        annotated_frame = self.box_annotator.annotate(scene=img.copy(), detections=detections)
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels)
        cv2.imwrite(output_image, annotated_frame)
        
    def get_croped_person(self, image_path, results, text):
        image = cv2.imread(image_path)
        input_boxes = results[0]
        class_ids = results[1]
        class_names = results[2]
        confidences = results[3]
        person_details = []
        
        for index in range(len(results[0])):
            if class_names[index] in ['rowboat', 'raft']:
                bbox = input_boxes[index]
                x_min, y_min, x_max, y_max = map(int, bbox)
                cropped_image = image[y_min:y_max, x_min:x_max]
                # cls_result = self.boat_model(cropped_image)[0]
                # cls_name = cls_result.names[cls_result.probs.top1]
                # if [cls_name] not in text:
                #     text.append([cls_name])
                # class_id = text.index([cls_name])
                # class_ids[index] = class_id 
                # print(f'{class_names[index]}->{cls_name}')
                # class_names[index] = cls_name
                # confidences[index] = float(cls_result.probs.top1conf.detach().cpu())
                # if cls_name == 'rowboat':
                    # detect person 
                person_result = self.inference(self.model, cropped_image, [['person'], [' ']], self.test_pipeline, score_thr=0.1)
                for j in range(len(person_result[0])):
                    person_result[0][j][0] += x_min
                    person_result[0][j][1] += y_min
                    person_result[0][j][2] += x_min
                    person_result[0][j][3] += y_min
                person_details.append(person_result)
        if ['person'] not in text and len(person_details):
            text.append(['person'])
        for i, sublist in enumerate(text):
            if 'person' in sublist:
                person_index = i
        for person_info in person_details:
            if person_info[0].shape[0]:
                input_boxes = np.concatenate([input_boxes, person_info[0]], axis=0)
                for j in range(len(person_info[2])): 
                    class_ids = np.append(class_ids, person_index)
                confidences = np.concatenate([confidences, person_info[3]], axis=0)
                class_names.extend(person_info[2])
        return [input_boxes, class_ids, class_names, confidences], text
        
    def yolo_cls_filter(self, image_path, results, text):
        image = cv2.imread(image_path)
        input_boxes = results[0]
        class_ids = results[1]
        class_names = results[2]
        confidences = results[3]
        person_details = []
        for index in range(len(results[0])):
            if class_names[index] in ['billowing black smoke', 'smoke']:
                bbox = input_boxes[index]
                x_min, y_min, x_max, y_max = map(int, bbox)
                cropped_image = image[y_min:y_max, x_min:x_max]
                cls_result = self.smoke_model(cropped_image)[0]
                cls_name = cls_result.names[cls_result.probs.top1]
                class_id = class_ids[index]
                if [cls_name] not in text:
                    text.append([cls_name])
                class_id = text.index([cls_name])
                class_ids[index] = class_id 
                print(f'{class_names[index]}->{cls_name}')
                class_names[index] = cls_name
                confidences[index] = float(cls_result.probs.top1conf.detach().cpu())
                
        #     if class_names[index] in ['boat']:
        #         bbox = input_boxes[index]
        #         x_min, y_min, x_max, y_max = map(int, bbox)
        #         cropped_image = image[y_min:y_max, x_min:x_max]
        #         cls_result = self.boat_model(cropped_image)[0]
        #         cls_name = cls_result.names[cls_result.probs.top1]
        #         if [cls_name] not in text:
        #             text.append([cls_name])
        #         class_id = text.index([cls_name])
        #         class_ids[index] = class_id 
        #         print(f'{class_names[index]}->{cls_name}')
        #         class_names[index] = cls_name
        #         confidences[index] = float(cls_result.probs.top1conf.detach().cpu())
        #         if cls_name == 'rowboat':
        #             # detect person 
        #             person_result = self.inference(self.model, cropped_image, [['person'], [' ']], self.test_pipeline, score_thr=0.1)
        #             for j in range(len(person_result[0])):
        #                 person_result[0][j][0] += x_min
        #                 person_result[0][j][1] += y_min
        #                 person_result[0][j][2] += x_min
        #                 person_result[0][j][3] += y_min

        #             person_details.append(person_result)
        # if ['person'] not in text and len(person_details):
        #     text.append(['person'])
        # for i, sublist in enumerate(text):
        #     if 'person' in sublist:
        #         person_index = i
        # for person_info in person_details:
        #     if person_info[0].shape[0]:
        #         input_boxes = np.concatenate([input_boxes, person_info[0]], axis=0)
        #         for j in range(len(person_info[2])): 
        #             class_ids = np.append(class_ids, person_index)
        #         confidences = np.concatenate([confidences, person_info[3]], axis=0)
        #         class_names.extend(person_info[2])
        return [input_boxes, class_ids, class_names, confidences], text
        
        
    def process_image(self, image_path, text_str, output_root, task, score_thr=0.05, show_result=False):
        image_name = image_path.split('/')[-1].split('.')[0]
        # output_root = os.path.join(output_path, image_name)
        os.makedirs(output_root, exist_ok=True)
        output_result = os.path.join(output_root, image_name+'_yolo_detection.json')
        text = []
        prompts = text_str.split('.')
        for prompt in prompts[:-1]:
            text.append([prompt])
        text.append([' '])
        results = self.inference(self.model, image_path, text, self.test_pipeline, score_thr=score_thr)
        # if task in ['black_smoke']:
        #     results, text = self.yolo_cls_filter(image_path, results, text)
        # if task in ['boat']:
        #     results, text = self.get_croped_person(image_path, results, text)
        input_boxes = results[0]
        class_ids = results[1]
        class_names = results[2]
        confidences = results[3]
        
        class_names_len = len(class_names)
        class_names = [c.lower() for c in class_names]
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]
        score_threshold = 0.3
        detections = sv.Detections(
            xyxy=input_boxes,
            class_id=class_ids,
            confidence=np.array(confidences),
        )  
        # filtered_detections = sv.Detections(
        #     xyxy=detections.xyxy[detections.confidence > score_threshold],
        #     confidence=detections.confidence[detections.confidence > score_threshold],
        #     class_id=detections.class_id[detections.confidence > score_threshold]
        # )
        # detections = filtered_detections
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
            # results, text = self.yolo_cls_filter(image_path, results, text)
        
        class_ids = detections.class_id
        confidences = detections.confidence
        class_names = [text[i][0] for i in class_ids]
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]
        # if show_result:
        #     output_image = os.path.join(output_root, image_name+'_yolo.png')
        #     self.show_bbox(image_path, detections, labels, output_image)
        outputs = {
            'image_path': image_path,
            'xyxy': detections.xyxy.tolist(),
            'confidence': detections.confidence.tolist(),
            'class_name': class_names,
        }
        # with open(output_result, 'w') as f:
        #     json.dump(outputs, f)
        return output_result, outputs
    
    def inference(self, model, image, texts, test_pipeline, score_thr=0.05, max_dets=100):
        if type(image) == str:
            image = cv2.imread(image)
        image = image[:, :, [2, 1, 0]]
        data_info = dict(img=image, img_id=0, texts=texts)
        data_info = test_pipeline(data_info)
        data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                        data_samples=[data_info['data_samples']])
        with torch.no_grad():
            output = model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        # score thresholding
        pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
        # max detections
        if len(pred_instances.scores) > max_dets:
            indices = pred_instances.scores.float().topk(max_dets)[1]
            pred_instances = pred_instances[indices]

        pred_instances = pred_instances.cpu().numpy()
        boxes = pred_instances['bboxes']
        labels = pred_instances['labels']
        scores = pred_instances['scores']
        label_texts = [texts[x][0] for x in labels]
        return boxes, labels, label_texts, scores

def process_task(model, tasks_config):
    '''
    use for process task:
    configs[dict]:  input config dict 
    task[str]: input processed task name 
    model[dict]: input grounded dino and sam2 models 
    '''
    ext = tasks_config['ext']
    data_path = tasks_config['data_path']
    test_images = glob.glob(os.path.join(data_path, f'*.{ext}'))
    prompt = tasks_config['prompt']
    tqbar = tqdm.tqdm(iterable=len(test_images), desc=f'Run in {data_path} images')
    for image in test_images:
        model.process_image(image, prompt, tasks_config['output_path'], show_result=True)
        tqbar.update(1)
        
yolo_config_file = './yolo_server_config.yaml'
yolo_det = YOLO_WORLD(yolo_config_file)
# with open('./batch_test_config.yaml', 'r') as f:
#     configs = yaml.load(f, Loader=yaml.Loader)
app = Flask(__name__)
@app.route('/yolo_detection', methods=['POST'])
def yolo_detection():
    start_time = time.time()
    data = request.get_json()
    image_path = data['image_path']
    task = data['task']
    output_root = os.path.dirname(image_path)
    output_root = data.get('output_root', output_root)
    text_prompt = data.get('text_prompt', '.')
    score_thr = float(data.get('score_thr', 0.05))
    output_file, outputs = yolo_det.process_image(image_path, text_prompt, output_root, task=task,score_thr=score_thr, show_result=True)
    outputs, annote_image = yolo_det.post_processer.post_task(task, outputs)
    with open(output_file, 'w') as f:
        json.dump(outputs, f)
    cv2.imwrite(output_file.replace('.json', '.jpg'), annote_image)
    response = {}
    response['output_file'] = output_file
    response['outputs'] = outputs
    end_time = time.time()
    use_time = round(end_time - start_time, 3)
    print(use_time)
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
# if __name__ == "__main__":
#     with open('./batch_test_config.yaml', 'r') as f:
#         configs = yaml.load(f, Loader=yaml.Loader)
#     yolo_config_file = './yolo_server_config.yaml'
#     task = 'fire'
#     task_config = configs[task]
#     yolo_det = YOLO_WORLD(yolo_config_file)
#     # process_task(yolo_det, task_config)
#     image_path = '/mnt/fillipo/yaowei/tower/benchmark/fire/bad_case/202401013860970496.jpg'
#     output_root = '/mnt/fillipo/yaowei/tower/test_output/bad_case/202401013860970496'
#     text_prompt = 'smoke.fire.chimneys.person.farmland.factory.building.forest.mountain.'
#     output_file, outputs = yolo_det.process_image(image_path, text_prompt, output_root, score_thr=0.07,show_result=True)
  