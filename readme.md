# ymir-live-code-executor demo

- [live-code-executor](https://github.com/IndustryEssentials/ymir-executor/live-code-executor) 支持在线运行的ymir镜像
- [ymir镜像数据传输接口](https://github.com/IndustryEssentials/ymir/blob/master/docs/ymir-cmd-container.md)
- [ymir-remote-git](https://github.com/IndustryEssentials/ymir-remote-git) 支持ymir镜像在线运行的代码库
- [ymir-executor](https://github.com/IndustryEssentials/ymir-executor) ymir离线运行镜像

| docker image | git-branch | task | result |
| - | - | - | - |
| youdaoyzbx/ymir-executor:live-code-base-tmi | base | training/mining/infer | ✔️/✔️/✔️ |
| youdaoyzbx/ymir-executor:live-code-base-tmi | ymir-remote-v7.0 | training/mining/infer | ✔️/✔️/✔️ |

## 基本命名

- `live-code-executor`： 可以根据git_url和git_branch进行代码拉取，依赖安装并运行的镜像
- `ymir-remote-git`： git_url和git_branch对应的代码
- `code_config`与`executor_config`： `executor_config`包含ymir预定义的keys与用户通过ymir-web端定义的keys, 而`code_config`
从`ymir-remote-git`代码中获得，具体哪一个文件可由ymir-web端定义的"code_config"字段确定。

## 镜像运行说明

- ymir通过nvidia-docker启动`live-code-executor`，并挂载`/in`和`/out`目录来处理镜像的输入和输出
- `live-code-executor`将拉代码，装依赖，并运行`ymir-remote-git`中的start.py
- `live-code-executor`通过预安装ymir-exc来读取ymir配置，读取输入的图片和标注，同时ymir-exc可以将返回权重，指标等写到ymir系统

## 分支说明
- ymir-remote-git/base: 简单演示`ymir-remote-git`的接口，不具有实际训练，挖掘与推理能力
    - configs/a.yaml: "code_config"文件
    - configs/b.yaml: "code_config"文件
    - extra-requirements.txt: python 依赖文件
    - start.py: 训练/挖掘/推理的启动程序

- ymir-remote-git/ymir-remote-v7.0: 以yolov5-v7.0版本为基础，演示`ymir-remote-git`的训练，挖掘与推理功能
    - configs/default.yaml: "code_config”文件
    - utils/ymir_yolov5.py: ymir所需的一些功能函数
    - train.py: 在yolov5框架上进行一些保存路径修改，ymir进度汇报等
    - mining/data_augment.py: 挖掘算法所需的数据增加功能函数
    - mining/mining_cald.py: 挖掘算法主程序
    - start.py：训练/挖掘/推理的启动程序
