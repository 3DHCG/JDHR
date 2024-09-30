# JDHR

## Introduction (介绍)

JDHR是一个基于Jittor国产深度学习框架的动态人体渲染算法库。该算法库全面集成了动态人体渲染的关键技术，包括点云采样、4D特征网格表示以及实时渲染等多个关键模块。

JDHR (Jittor-based Dynamic Human Rendering) is a dynamic human rendering algorithm library based on Jittor. This algorithm library fully integrates key technologies, including point cloud sampling, 4D feature grid representation, and real-time rendering.

## Plan (开源计划)
- [x] **Release training code**
- [ ] **Release realtime rendering code**
- [ ] **Release initializing point clouds code**
- [ ] **Build a  WebSocket-based viewer**

## Installation（安装）

Install the basic environment under the JDHR repo:

```shell
# Editable install, with dependencies from requirements.txt
pip install -v -e . 
```

Install Rasterizer for realtime rendering:

```shell
cd easyvolcap/diff_point_rasterizater
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CUDA_ARCHITECTURES=86(根据显卡版本选用70.75.86)
make -j4
```

## Datasets（数据集）

### DNA-Rendering Datasets

Please refer to [HumanRF](https://github.com/synthesiaresearch/humanrf) to download DNA-Rendering datasets.
Note that you should cite the corresponding papers if you use these datasets.

### :clock3: more datasets

## Training（训练）

This script trains a single-frame version on the first frame of the *0013_09* sequence of the *DNA-Rendering* dataset. You can quickly verify whether your dataset preparation and installation process are correct.

```shell
evc-train -c configs/exps/4k4d/4k4d_0013_09_r4.yaml,configs/specs/static.yaml,configs/specs/tiny.yaml exp_name=4k4d_0013_09_r4_static
```

The actual training of the full model is more straight forward:

```shell
evc-train -c configs/exps/4k4d/4k4d_0013_09_r4.yaml
```

## :clock3:Rendering（渲染）

## Team （团队）

动态人体渲染算法库（JDHR）是由清华大学和北京交通大学团队共同维护的开源代码库。欢迎大家使用JDHR开展研究工作。

JDHR is an open-source code repository jointly maintained by teams from Tsinghua University and Beijing Jiaotong University

Feel free to request support for new models and contribute to JDHR.


## Acknowledgement (鸣谢)
1. [Jittor](https://github.com/Jittor/jittor)
2. [JNeRF](https://github.com/Jittor/JNeRF)
3. [EasyVolcap](https://github.com/zju3dv/EasyVolcap)
4. [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)
