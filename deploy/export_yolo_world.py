# # Copyright (c) OpenMMLab. All rights reserved.
import os
import json
import warnings
import argparse
from io import BytesIO

import onnx
import torch
import yaml
from mmdet.apis import init_detector
from mmengine.config import ConfigDict
from mmengine.logging import print_log
from mmengine.utils.path import mkdir_or_exist

from easydeploy.model import DeployModel, MMYOLOBackend  # noqa E402

warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)
warnings.filterwarnings(action='ignore', category=torch.jit.ScriptWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=ResourceWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--custom-text',
                        type=str,
                        help='custom text inputs (text json) for YOLO-World.')
    parser.add_argument('--add-padding',
                        action="store_true",
                        help="add an empty padding to texts.")
    parser.add_argument('--model-only',
                        action='store_true',
                        help='Export model only')
    parser.add_argument('--without-nms',
                        action='store_true',
                        help='Export model without NMS')
    parser.add_argument('--without-bbox-decoder',
                        action='store_true',
                        help='Export model without Bbox Decoder (for INT8 Quantization)')
    parser.add_argument('--work-dir',
                        default='./work_dirs',
                        help='Path to save export model')
    parser.add_argument('--img-size',
                        nargs='+',
                        type=int,
                        default=[640, 640],
                        help='Image size of height and width')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--device',
                        default='cuda:0',
                        help='Device used for inference')
    parser.add_argument('--simplify',
                        action='store_true',
                        help='Simplify onnx model by onnx-sim')
    parser.add_argument('--opset',
                        type=int,
                        default=11,
                        help='ONNX opset version')
    parser.add_argument('--backend',
                        type=str,
                        default='onnxruntime',
                        help='Backend for export onnx')
    parser.add_argument('--pre-topk',
                        type=int,
                        default=1000,
                        help='Postprocess pre topk bboxes feed into NMS')
    parser.add_argument('--keep-topk',
                        type=int,
                        default=100,
                        help='Postprocess keep topk bboxes out of NMS')
    parser.add_argument('--iou-threshold',
                        type=float,
                        default=0.65,
                        help='IoU threshold for NMS')
    parser.add_argument('--score-threshold',
                        type=float,
                        default=0.04,
                        help='Score threshold for NMS')
    args = parser.parse_args()
    args.img_size *= 2 if len(args.img_size) == 1 else 1
    return args


def build_model_from_cfg(config_path, checkpoint_path, device):
    model = init_detector(config_path, checkpoint_path, device=device)
    model.eval()
    return model


def main():
    config_file = '../config.yaml'
    with open(config_file, 'r') as f:
        configs = yaml.load(f, Loader=yaml.Loader)
    model_config_file = configs['yolo_server']['config_file']
    checkpoints = configs['yolo_server']['checkpoint']
    save_onnx_path = checkpoints.replace('.pth', '.onnx')
    device = 'cuda:0'
    backend = MMYOLOBackend('onnxruntime')
    postprocess_cfg = ConfigDict(pre_top_k=1000,
                                 keep_top_k=100,
                                 iou_threshold=0.7,
                                 score_threshold=0.3)
    output_names = ['num_dets', 'boxes', 'scores', 'labels']
    prompt = configs['prompt']
    texts = prompt.split('.') 
    texts += [' ']
    baseModel = build_model_from_cfg(model_config_file, checkpoints, device)
    if hasattr(baseModel, 'reparameterize'):
        # reparameterize text into YOLO-World
        baseModel.reparameterize([texts])

    deploy_model = DeployModel(baseModel=baseModel,
                               backend=backend,
                               postprocess_cfg=postprocess_cfg,
                               with_nms=False,
                               without_bbox_decoder=True)
    deploy_model.eval()
    fake_input = torch.randn(1, 3, 640).to(device)
    deploy_model(fake_input)
    with BytesIO() as f:
        torch.onnx.export(deploy_model,
                          fake_input,
                          f,
                          input_names=['images'],
                          output_names=output_names,
                          opset_version=12)
        f.seek(0)
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)
        # Fix tensorrt onnx output shape, just for view
        # if not args.model_only and not args.without_nms and backend in (
        #         MMYOLOBackend.TENSORRT8, MMYOLOBackend.TENSORRT7):
        #     shapes = [
        #         args.batch_size, 1, args.batch_size, args.keep_topk, 4,
        #         args.batch_size, args.keep_topk, args.batch_size,
        #         args.keep_topk
        #     ]
        #     for i in onnx_model.graph.output:
        #         for j in i.type.tensor_type.shape.dim:
        #             j.dim_param = str(shapes.pop(0))
    # if args.simplify:
    #     try:
    #         import onnxsim
    #         onnx_model, check = onnxsim.simplify(onnx_model)
    #         assert check, 'assert check failed'
    #     except Exception as e:
    #         print_log(f'Simplify failure: {e}')
    onnx.save(onnx_model, save_onnx_path)
    print_log(f'ONNX export success, save into {save_onnx_path}')


if __name__ == '__main__':
    main()
