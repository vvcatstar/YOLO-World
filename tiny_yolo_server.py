import os
import glob
import json
import yaml
import os.path as osp
import cv2
import numpy as np
import supervision as sv
import supervision as sv
import tqdm
import time
from ultralytics import YOLO
from flask import Flask, request, jsonify
from IPython import embed 
from post_process import PostProcess
class YOLO_WORLD:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.configs = yaml.load(f, Loader=yaml.Loader)
        self.post_processer = PostProcess(self.configs)
        self.model = YOLO(self.configs['post_config']['person_det'])

    def process_image(self, image_path, text_str, output_root, task, score_thr=0.05, show_result=False, bboxInfos=[]):
        image_name = image_path.split('/')[-1].split('.')[0]
        os.makedirs(output_root, exist_ok=True)
        output_result = os.path.join(output_root, image_name+'_yolo_detection.json')
        text = []
        prompts = text_str.split('.')
        for prompt in prompts[:-1]:
            text.append([prompt])
        text.append([' '])
        self.model.set_classes(prompts)
        results = self.inference(self.model, image_path, text,score_thr=score_thr)
        input_boxes = results[0]
        class_ids = results[1]
        class_names = results[2]
        confidences = results[3]
        class_names = [c.lower() for c in class_names]
        if len(class_names):
            detections = sv.Detections(
                xyxy=np.array(input_boxes),
                class_id=np.array(class_ids),
                confidence=np.array(confidences),
            )  
            detections = detections.with_nms(threshold=0.5, class_agnostic=True)
            class_ids = detections.class_id
            confidences = detections.confidence
            class_names = [text[i][0] for i in class_ids]
            outputs = {
                'image_path': image_path,
                'xyxy': detections.xyxy.tolist(),
                'confidence': detections.confidence.tolist(),
                'class_name': class_names,
            }
        else:
            outputs = {
                'image_path': image_path,
                'xyxy': [],
                'confidence': [],
                'class_name': [],
            }
        if bboxInfos:
            for bboxInfo in bboxInfos:
                h, w = cv2.imread(image_path).shape[:2]
                conf = bboxInfo['confidenceLevel']
                point_list = bboxInfo['alarmPointList']
                points = []
                for point in point_list:
                    points.append((point['percentX'], point['percentY']))
                points = np.array(points)
                y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
                x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
                anno_bbox = [x_min*w, y_min*h, x_max*w, y_max*h]
                anno_label = bboxInfo['label']
                outputs['class_name'].append(anno_label)
                outputs['xyxy'] = np.vstack([outputs['xyxy'], anno_bbox])
                outputs['confidence'].append(conf)
        return output_result, outputs
    
    def inference(self, model, image, texts, score_thr=0.05, max_dets=100):
        pred_instances = model(image, conf=score_thr, iou=0.45)[0]
        results = json.loads(pred_instances.to_json())
        boxes = [[result['box']['x1'], result['box']['y1'], result['box']['x2'], result['box']['y2']] for result in results]
        labels = [result['name'] for result in results]
        scores = [result['confidence'] for result in results]
        class_id = [result['class'] for result in results]
        return boxes, class_id, labels, scores

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
        
yolo_config_file = '../config.yaml'
yolo_det = YOLO_WORLD(yolo_config_file)
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
    bboxInfos = data.get('bboxInfos', [])
    output_file, outputs = yolo_det.process_image(image_path, text_prompt, output_root, task=task,score_thr=score_thr, show_result=True, bboxInfos=bboxInfos)
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
  