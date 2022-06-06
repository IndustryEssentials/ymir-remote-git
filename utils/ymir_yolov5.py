"""
function for combine ymir and yolov5
"""
import os
import os.path as osp
import shutil
from typing import List, Dict, Tuple, Any

import imagesize
import numpy as np
import torch
import yaml
from nptyping import UInt8, NDArray, Shape
from ymir_exc import dataset_reader as dr
from ymir_exc import env, monitor
from ymir_exc import result_writer as rw

from models.common import DetectMultiBackend
from models.experimental import attempt_download
from utils.augmentations import letterbox
from utils.dataloaders import img2label_paths
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

# const value for process
PREPROCESS_PERCENT = 0.1
TASK_PERCENT = 0.8

CV_IMAGE = NDArray[Shape['*,*,3'], UInt8]
BBOX = NDArray[Shape['*,4'], Any]


def get_code_config() -> dict:
    executor_config = env.get_executor_config()
    code_config_file = executor_config.get('code_config', None)

    if code_config_file is None:
        return dict()
    else:
        with open(code_config_file, 'r') as f:
            return yaml.safe_load(f)


def get_merged_config() -> dict:
    """
    merge executor_config and code_config
    code_config will be overwrited.
    """
    exe_cfg = env.get_executor_config()
    code_cfg = get_code_config()
    code_cfg.update(exe_cfg)
    return code_cfg


def get_weight_file(try_download: bool = True) -> str:
    """
    return the weight file path by priority

    1. executor_config['model_params_path'] or executor_config['model_params_path']
    2. if try_download and no weight file offered
            for training task, yolov5 will download it from github.
    """
    executor_config = get_merged_config()
    ymir_env = env.get_current_env()

    env_config = env.get_current_env()
    if env_config.run_training:
        model_params_path = executor_config['pretrained_model_paths']
    else:
        model_params_path = executor_config['model_params_path']

    model_dir = osp.join(ymir_env.input.root_dir,
                         ymir_env.input.models_dir)
    model_params_path = [p for p in model_params_path if osp.exists(osp.join(model_dir, p))]

    # choose weight file by priority, best.pt > xxx.pt
    if 'best.pt' in model_params_path:
        return osp.join(model_dir, 'best.pt')
    else:
        for f in model_params_path:
            if f.endswith('.pt'):
                return osp.join(model_dir, f)

    # if no weight file offered
    if try_download and env_config.run_training:
        model_name = executor_config['model']
        weights = attempt_download(f'{model_name}.pt')
        return weights
    elif try_download and not env_config.run_training:
        # donot allow download weight for mining and infer
        raise Exception('try_download is allowed for training task only ' +
                        'please offer model weight in executor_config')
    elif env_config.run_training:
        # no pretrained weight
        return ""
    else:
        raise Exception(f'no weight file offered in {model_dir} ' +
                        'please offer model weight in executor_config')


class YmirYolov5():
    """
    used for mining and inference to init detector and predict.
    """

    def __init__(self):
        executor_config = get_merged_config()
        gpu_id = executor_config.get('gpu_id', '0')
        gpu_num = len(gpu_id.split(','))
        if gpu_num == 0:
            device = 'cpu'
        else:
            device = gpu_id
        device = select_device(device)

        self.model = self.init_detector(device)
        self.device = device
        self.class_names = executor_config['class_names']
        self.stride = self.model.stride
        self.conf_thres = float(executor_config['conf_thres'])
        self.iou_thres = float(executor_config['iou_thres'])

        img_size = int(executor_config['img_size'])
        imgsz = (img_size, img_size)
        imgsz = check_img_size(imgsz, s=self.stride)
        # Run inference
        self.model.warmup(imgsz=(1, 3, *imgsz))  # warmup

        self.img_size = imgsz

    def init_detector(self, device: torch.device) -> DetectMultiBackend:
        weights = get_weight_file(try_download=False)

        model = DetectMultiBackend(weights=weights,
                                   device=device,
                                   dnn=False,  # not use opencv dnn for onnx inference
                                   data='data.yaml')  # dataset.yaml path

        return model

    def predict(self, img: CV_IMAGE) -> NDArray:
        """
        predict single image and return bbox information
        img: opencv BGR, uint8 format
        """
        # preprocess: padded resize
        img1 = letterbox(img, self.img_size, stride=self.stride, auto=True)[0]

        # preprocess: convert data format
        img1 = img1.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img1 = np.ascontiguousarray(img1)
        img1 = torch.from_numpy(img1).to(self.device)

        img1 = img1 / 255  # 0 - 255 to 0.0 - 1.0
        img1.unsqueeze_(dim=0)  # expand for batch dim
        pred = self.model(img1)

        # postprocess
        conf_thres = self.conf_thres
        iou_thres = self.iou_thres
        classes = None  # not filter class_idx in results
        agnostic_nms = False
        max_det = 1000

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        result = []
        for det in pred:
            if len(det):
                # Rescale boxes from img_size to img size
                det[:, :4] = scale_coords(img1.shape[2:], det[:, :4], img.shape).round()
                result.append(det)

        # xyxy, conf, cls
        if len(result) > 0:
            tensor_result = torch.cat(result, dim=0)
            numpy_result = tensor_result.data.cpu().numpy()
        else:
            numpy_result = np.zeros(shape=(0, 6), dtype=np.float32)

        return numpy_result

    def infer(self, img: CV_IMAGE) -> List[rw.Annotation]:
        anns = []
        result = self.predict(img)

        for i in range(result.shape[0]):
            xmin, ymin, xmax, ymax, conf, cls = result[i, :6].tolist()
            ann = rw.Annotation(class_name=self.class_names[int(cls)], score=conf, box=rw.Box(
                x=int(xmin), y=int(ymin), w=int(xmax - xmin), h=int(ymax - ymin)))

            anns.append(ann)

        return anns


def digit(x: int) -> int:
    """
    get the filename length
    x: the number of images
    return the numerical digit
    """
    return len(str(x))


def convert_ymir_to_yolov5(output_root_dir: str) -> None:
    """
    convert ymir format dataset to yolov5 format
    output_root_dir: the output root dir
    """
    os.makedirs(output_root_dir, exist_ok=True)
    os.makedirs(osp.join(output_root_dir, 'images'), exist_ok=True)
    os.makedirs(osp.join(output_root_dir, 'labels'), exist_ok=True)

    ymir_env = env.get_current_env()

    if ymir_env.run_training:
        train_data_size = dr.items_count(env.DatasetType.TRAINING)
        val_data_size = dr.items_count(env.DatasetType.VALIDATION)
        N = len(str(train_data_size + val_data_size))
        splits = ['train', 'val']
    elif ymir_env.run_mining:
        N = dr.items_count(env.DatasetType.CANDIDATE)
        splits = ['test']
    elif ymir_env.run_infer:
        N = dr.items_count(env.DatasetType.CANDIDATE)
        splits = ['test']

    idx = 0
    DatasetTypeDict = dict(train=env.DatasetType.TRAINING,
                           val=env.DatasetType.VALIDATION,
                           test=env.DatasetType.CANDIDATE)

    digit_num = digit(N)
    monitor_gap = max(1, N // 10)
    for split in splits:
        split_imgs = []
        for asset_path, annotation_path in dr.item_paths(dataset_type=DatasetTypeDict[split]):
            idx += 1
            if idx % monitor_gap == 0:
                monitor.write_monitor_logger(percent=PREPROCESS_PERCENT * idx / N)

            # only copy image for training task
            # valid train.txt, val.txt and invalid test.txt for training task
            # invalid train.txt, val.txt and test.txt for infer and mining task
            if split in ['train', 'val']:
                img_suffix = osp.splitext(asset_path)[1]
                img_path = osp.join(output_root_dir, 'images', str(idx).zfill(digit_num) + img_suffix)
                shutil.copy(asset_path, img_path)
                ann_path = osp.join(output_root_dir, 'labels', str(idx).zfill(digit_num) + '.txt')
                yolov5_ann_path = img2label_paths([img_path])[0]

                if yolov5_ann_path != ann_path:
                    raise Exception(f'bad yolov5_ann_path={yolov5_ann_path} and ann_path = {ann_path}')

                width, height = imagesize.get(img_path)

                # convert annotation_path to ann_path
                with open(ann_path, 'w') as fw:
                    with open(annotation_path, 'r') as fr:
                        for line in fr.readlines():
                            class_id, xmin, ymin, xmax, ymax = [int(x) for x in line.strip().split(',')]

                            # class x_center y_center width height
                            # normalized xywh
                            # class_id 0-indexed
                            xc = (xmin + xmax) / 2 / width
                            yc = (ymin + ymax) / 2 / height
                            w = (xmax - xmin) / width
                            h = (ymax - ymin) / height
                            fw.write(f'{class_id} {xc} {yc} {w} {h}\n')

                split_imgs.append(img_path)
        if split in ['train', 'val']:
            with open(osp.join(output_root_dir, f'{split}.txt'), 'w') as fw:
                fw.write('\n'.join(split_imgs))

    # generate data.yaml for training/mining/infer
    config = env.get_executor_config()
    data = dict(path=output_root_dir,
                train="train.txt",
                val="val.txt",
                test='test.txt',
                nc=len(config['class_names']),
                names=config['class_names'])

    with open('data.yaml', 'w') as fw:
        fw.write(yaml.safe_dump(data))


def write_ymir_training_result(results: Tuple, maps: NDArray, rewrite=False) -> int:
    """
    results: (mp, mr, map50, map, loss)
    maps: map@0.5:0.95 for all classes
    rewrite: set true to ensure write the best result
    """
    if not rewrite:
        training_result_file = env.get_current_env().output.training_result_file
        if osp.exists(training_result_file):
            return 0

    executor_config = get_merged_config()
    model = executor_config['model']
    class_names = executor_config['class_names']
    mp = results[0]  # mean of precision
    mr = results[1]  # mean of recall
    map50 = results[2]  # mean of ap@0.5
    map = results[3]  # mean of ap@0.5:0.95

    # use `rw.write_training_result` to save training result
    rw.write_training_result(model_names=[f'{model}.yaml', 'best.pt', 'last.pt', 'best.onnx'],
                             mAP=float(map),
                             mAP50=float(map50),
                             precision=float(mp),
                             recall=float(mr),
                             classAPs={class_name: v
                                       for class_name, v in zip(class_names, maps.tolist())})
    return 0
