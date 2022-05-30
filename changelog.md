# change log

## 2022-05-30

adapt yolov5 to ymir live code executor

- 添加mining文件夹，为yolov5添加挖掘功能。其中mining/data_augment.py进行数据增强，mining/mining_cald.py实现挖掘算法。在start.py的推理任务中将会通过subprocess调用mining/mining_cald.py
- 添加configs文件夹，添加配置文件configs/default.yaml
- 添加utils/ymir_yolov5.py，提供训练/挖掘/推理任务所需要的一些函数。
- 添加start.py, 远端代码执行镜像将会调用start.py。
- 修改train.py，修改yolov5权重，tensorboard日志的保存目录，回调monitor进度。