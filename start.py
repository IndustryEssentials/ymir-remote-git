import logging
import os
import shutil
import sys
import time
from typing import List

import torch
import torchvision
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm
from ymir_exc import dataset_reader as dr
from ymir_exc import env, monitor
from ymir_exc import result_writer as rw


def get_code_config() -> dict:
    executor_config = env.get_executor_config()
    code_config_file = executor_config.get('code_config', None)

    try:
        remote_config = dict(git_url=executor_config['git_url'],
                             git_branch=executor_config.get(
                                 'git_branch', 'master'),
                             code_config=code_config_file)

        # if code_config_file == '' or code_config_file is None:
        if not code_config_file:
            return dict()
        else:
            with open(code_config_file, 'r') as f:
                return yaml.safe_load(f)
    except Exception as e:
        logging.error(f'remote config {remote_config} with error: {e}')
        raise


def get_merged_config() -> dict:
    """
    merge executor_config and code_config
    """

    # exe_cfg overwrite code_cfg
    exe_cfg = env.get_executor_config()
    code_cfg = get_code_config()

    exe_cfg.update(code_cfg)

    return exe_cfg


def start() -> int:
    env_config = env.get_current_env()

    if env_config.run_training:
        _run_training(env_config)
    elif env_config.run_mining:
        _run_mining(env_config)
    elif env_config.run_infer:
        _run_infer(env_config)
    else:
        logging.info('no task to run!!!')

    return 0


def _run_training(env_config: env.EnvConfig) -> None:
    """
    sample function of training, which shows:
    1. how to get config
    2. how to read training and validation datasets
    3. how to write logs
    4. how to write training result
    """
    # use `env.get_merged_config` to get config file for training
    executor_config = get_merged_config()
    class_names: List[str] = executor_config['class_names']
    expected_mAP: float = executor_config.get('map', 0.6)
    epoch: int = executor_config.get('epoch', 10)
    model: str = executor_config.get('model', 'vgg11')

    config = edict(epoch=epoch)

    # use `logging` or `print` to write log to console
    #   notice that logging.basicConfig is invoked at executor.env
    logging.info(f"training config: {executor_config}")

    # read training dataset items
    # note that `dataset_reader.item_paths` is a generator
    for asset_path, ann_path in dr.item_paths(env.DatasetType.TRAINING):
        logging.info(f"asset: {asset_path}, annotation: {ann_path}")
        with open(ann_path, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                class_id, x1, y1, x2, y2 = [int(s)
                                            for s in line.strip().split(',')]
                name = class_names[class_id]
                logging.info(f"{name} xmin={x1} ymin={y1} xmax={x2} ymax={y2}")
        break

    # write task process percent to monitor.txt
    monitor.write_monitor_logger(percent=0.0)

    # fake training function
    _dummy_work(config)

    # suppose we have a long time training, and have saved the final model
    # use `env_config.output.models_dir` to get model output dir
    if model == 'vgg11':
        m = torchvision.models.vgg11(pretrained=False)
    else:
        m = torchvision.models.vgg13(pretrained=False)

    models_dir = env_config.output.models_dir
    os.makedirs(models_dir, exist_ok=True)
    torch.save(m, os.path.join(models_dir, f'{model}.pt'))

    # write other information
    with open(os.path.join(models_dir, 'model.yaml'), 'w') as f:
        f.write(f'model: {model}')
    shutil.copy('models/vgg.py',
                os.path.join(models_dir, 'vgg.py'))

    # use `rw.write_training_result` to save training result
    # the files in model_names will be saved and can be download from ymir-web
    rw.write_training_result(model_names=[f'{model}.pt',
                                          'model.yaml',
                                          'vgg.py'],
                             mAP=expected_mAP,
                             classAPs={class_name: expected_mAP
                                       for class_name in class_names})

    # if task done, write 100% percent log
    monitor.write_monitor_logger(percent=1.0)


def _run_mining(env_config: env.EnvConfig) -> None:
    # use `env.get_merged_config` to get config file for training
    executor_config = get_merged_config()
    epoch: int = executor_config.get('epoch', 10)

    weight_files = executor_config['model_params_path']
    models_dir = env_config.input.models_dir
    weight_files = [os.path.join(models_dir, f) for f in weight_files]
    logging.info(f'use weight files {weight_files}')

    config = edict(epoch=epoch)
    # use `logging` or `print` to write log to console
    logging.info(f"mining config: {executor_config}")

    # use `dataset_reader.item_paths` to read candidate dataset items
    #   note that annotations path will be empty str if no annotations
    asset_paths = []
    for asset_path, _ in dr.item_paths(env.DatasetType.CANDIDATE):
        asset_paths.append(asset_path)

    if len(asset_paths) == 0:
        raise ValueError('empty asset paths')

    # use `monitor.write_monitor_logger` to write task process to monitor.txt
    logging.info(f"assets count: {len(asset_paths)}")
    monitor.write_monitor_logger(percent=0.0)

    # fake mining function
    _dummy_work(config)

    # write mining result
    #   here we give a fake score to each assets
    total_length = len(asset_paths)
    mining_result = [(asset_path, index / total_length)
                     for index, asset_path in enumerate(asset_paths)]
    rw.write_mining_result(mining_result=mining_result)

    # if task done, write 100% percent log
    logging.info('mining done')
    monitor.write_monitor_logger(percent=1.0)


def _run_infer(env_config: env.EnvConfig) -> None:
    # use `env.get_merged_config` to get config file for training
    executor_config = get_merged_config()
    class_names: List[str] = executor_config['class_names']
    epoch: int = executor_config.get('epoch', 10)

    weight_files = executor_config['model_params_path']
    models_dir = env_config.input.models_dir
    weight_files = [os.path.join(models_dir, f) for f in weight_files]
    logging.info(f'use weight files {weight_files}')

    config = edict(epoch=epoch)
    # use `logging` or `print` to write log to console
    logging.info(f"infer config: {executor_config}")

    # use `dataset_reader.item_paths` to read candidate dataset items
    # note that annotations path will be empty str if no annotations
    asset_paths: List[str] = []
    for asset_path, _ in dr.item_paths(env.DatasetType.CANDIDATE):
        asset_paths.append(asset_path)

    logging.info(f"assets count: {len(asset_paths)}")

    if len(asset_paths) == 0 or len(class_names) == 0:
        raise ValueError('empty asset paths or class names')

    # write log to console and write task process percent to monitor.txt
    monitor.write_monitor_logger(percent=0.0)

    # fake infer function
    _dummy_work(config)

    # write infer result
    fake_annotation = rw.Annotation(
        class_name=class_names[0],
        score=0.9,
        box=rw.Box(x=50, y=50, w=150, h=150))
    infer_result = {asset_path: [fake_annotation]
                    for asset_path in asset_paths}
    rw.write_infer_result(infer_result=infer_result)

    # if task done, write 100% percent log
    logging.info('infer done')
    monitor.write_monitor_logger(percent=1.0)


def _dummy_work(config: edict) -> None:
    for e in tqdm(range(config.epoch)):
        time.sleep(1)
        monitor.write_monitor_logger(percent=e/config.epoch)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        format='%(levelname)-8s: [%(asctime)s] %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        level=logging.INFO)
    sys.exit(start())
