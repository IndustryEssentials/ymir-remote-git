import logging
import os.path as osp
import shutil
import subprocess
import sys

import cv2
from ymir_exc import dataset_reader as dr
from ymir_exc import env, monitor
from ymir_exc import result_writer as rw

from utils.ymir_yolov5 import (YmirYolov5, convert_ymir_to_yolov5,
                               get_weight_file, ymir_process_config,
                               get_universal_config)


def start() -> int:
    env_config = env.get_current_env()

    logging.add(osp.join(env_config.output.root_dir, 'ymir_start.log'))

    logging.info(f'env_config: {env_config}')

    if env_config.run_training:
        _run_training(env_config)
    elif env_config.run_mining:
        _run_mining(env_config)
    elif env_config.run_infer:
        _run_infer(env_config)
    else:
        logging.warning('no task running')

    return 0


def _run_training(env_config: env.EnvConfig) -> None:
    """
    function for training task
    1. get merged config from ymir_exc and code_config file
    2. convert dataset
    3. training model
    4. save model weight/hyperparameter/... to design directory
    """
    # 1. use `env.get_executor_config` to get config file for training
    executor_config = get_universal_config()

    # use `logging` or `print` to write log to console
    #   notice that logging.basicConfig is invoked at executor.env
    logging.info(f"training config: {executor_config}")

    # 2. convert dataset
    logging.info('convert ymir dataset to yolov5 dataset')
    out_dir = osp.join(env_config.output.root_dir, 'yolov5_dataset')
    convert_ymir_to_yolov5(out_dir)
    monitor.write_monitor_logger(percent=ymir_process_config['preprocess'])

    # 3. training model
    epochs = executor_config.get('epochs', 300)
    batch_size = executor_config.get('batch_size', 64)
    model = executor_config.get('model', 'yolov5s')
    img_size = executor_config.get('img_size', 640)
    weights = get_weight_file()

    models_dir = env_config.output.models_dir
    command = f'python train.py --epochs {epochs} ' + \
        f'--batch-size {batch_size} --data data.yaml --project /out ' + \
        f'--cfg models/{model}.yaml --name models --weights {weights} ' + \
        f'--img-size {img_size} --hyp data/hyps/hyp.scratch-low.yaml ' + \
        '--exist-ok'
    # use `monitor.write_monitor_logger` to write write task process percent to monitor.txt
    logging.info(f'start training: {command}')

    # os.system(command)
    subprocess.check_output(command.split())
    monitor.write_monitor_logger(percent=ymir_process_config['preprocess'] + ymir_process_config['task'])

    # 4. convert to onnx and save model weight to design directory
    opset = executor_config['opset']
    command = f'python export.py --weights {models_dir}/best.pt --opset {opset} --include onnx'
    logging.info(f'export onnx weight: {command}')
    subprocess.check_output(command.split())

    # save hyperparameter
    shutil.copy(f'models/{model}.yaml', f'{models_dir}/{model}.yaml')

    # if task done, write 100% percent log
    monitor.write_monitor_logger(percent=1.0)


def _run_mining(env_config: env.EnvConfig) -> None:
    executor_config = get_universal_config()
    logging.info(f"mining config: {executor_config}")

    logging.info('convert ymir dataset to yolov5 dataset')
    out_dir = osp.join(env_config.output.root_dir, 'yolov5_dataset')
    convert_ymir_to_yolov5(out_dir)
    monitor.write_monitor_logger(percent=ymir_process_config['preprocess'])

    command = 'python mining/mining_cald.py'
    logging.info(f'mining: {command}')
    subprocess.check_output(command.split())
    monitor.write_monitor_logger(percent=1.0)


def _run_infer(env_config: env.EnvConfig) -> None:
    executor_config = get_universal_config()
    logging.info(f"infer config: {executor_config}")

    # generate data.yaml for infer
    logging.info('convert ymir dataset to yolov5 dataset')
    out_dir = osp.join(env_config.output.root_dir, 'yolov5_dataset')
    convert_ymir_to_yolov5(out_dir)
    monitor.write_monitor_logger(percent=ymir_process_config['preprocess'])

    N = dr.items_count(env.DatasetType.CANDIDATE)
    infer_result = dict()
    model = YmirYolov5()
    idx = 0
    for asset_path, _ in dr.item_paths(dataset_type=env.DatasetType.CANDIDATE):
        img_path = osp.join(env_config.input.root_dir, env_config.input.assets_dir, asset_path)
        img = cv2.imread(img_path)

        result = model.infer(img)

        infer_result[asset_path] = result
        idx += 1
        monitor.write_monitor_logger(percent=ymir_process_config['preprocess'] + ymir_process_config['task'] * idx / N)

    rw.write_infer_result(infer_result=infer_result)
    monitor.write_monitor_logger(percent=1.0)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        format='%(levelname)-8s: [%(asctime)s] %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        level=logging.INFO)
    sys.exit(start())
