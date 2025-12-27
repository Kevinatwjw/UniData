# UniData 数据处理指南（一）：基于 ASAP 的 12Hz 高频插帧

**摘要**：本模块是 UniData 项目的数据基石。由于 nuScenes 原始数据集仅提供 2Hz 的关键帧标注，无法满足 UniScene 对 12Hz 连续视频和点云生成的训练需求。本指南将指导您集成并运行 **ASAP (Autonomous-driving StreAming Perception)** 算法，生成高质量的 12Hz 伪真值标注。

---

## 零、 项目背景与数据综述

### 0.1 ASAP 框架解析

[**ASAP**](https://github.com/JeffWang987/ASAP/tree/main) 旨在解决自动驾驶数据集中“传感器采样频率”与“人工标注频率”之间的巨大鸿沟。

在 nuScenes 中，LiDAR 采集频率为 20Hz，但官方仅提供 2Hz 的关键帧标注。这意味着 90% 的数据（中间帧/Sweeps）处于无监督状态。ASAP 提供了两种策略来填补这一空白：

1.  **基础插值策略 (Interp)**：简单线性插值，速度快，适合基准对比。
2.  **高级时序感知策略 (Advanced)**：核心贡献。利用预训练检测器（CenterPoint）对所有帧进行推理，构建时序关联，修正插值误差，生成高质量的高频标注。

### 0.2 数据集结构规范

ASAP 严格依赖 nuScenes 的官方目录结构。请确保您的数据存放路径符合以下规范：

```text
UniData/data/nuscenes/
├── maps/               # 高精地图数据
├── samples/            # [关键] 关键帧数据 (Keyframes)，官方提供标注 (2Hz)
│   ├── CAM_FRONT/
│   ├── LIDAR_TOP/
│   └── ...
├── sweeps/             # [关键] 中间帧数据 (Intermediate Frames)，无官方标注 (20Hz)
│   ├── CAM_FRONT/
│   ├── LIDAR_TOP/
│   └── ...
├── v1.0-mini/          # 包含各类元数据 JSON 文件
└── v1.0-trainval/      # 若使用完整集，则为此文件夹
```

---

## 一、 环境与数据准备

### 1.1 环境依赖
请确保您已按照 [环境搭建指南](../README.md) 完成了 `unidata` 虚拟环境的配置。本步骤核心依赖包括：
*   `Python 3.8`
*   `MMDetection3D v0.17.1`
*   `spconv-cu111`

### 1.2 脚本准备

ASAP 的处理流程依赖于几个核心 Shell 脚本。请到 `UniData/scripts/` 目录下查看。

> **注意**：ASAP 的核心 Python 代码位于项目根目录下的 **`sAP3D/`** 文件夹中。下述脚本中的 `PYTHONPATH` 设置已确保能正确调用该模块。

---

## 二、 完整执行流程与脚本详解

ASAP 的插帧过程分为四个严格顺序的步骤。

> **注意**：以下命令均以 **UniData 项目根目录** 为工作路径进行演示。

### 步骤 1：生成 20Hz 输入信息 (Input Generation)

**目标**：对原始数据进行线性插值，生成 20Hz 的基础元数据文件 (`.pkl`)。

**1. 准备脚本**： `scripts/nusc_20Hz_lidar_input_pkl.sh`

```bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# 1. 修改 data_path
# 注意：这里必须写到 'nuscenes' 这一层，不要写到 'v1.0-mini'
data_path="./data/nuscenes"

# 2. 修改 data_version
# 将默认的 trainval 改为 mini
data_version="v1.0-mini"
# data_version="v1.0-trainval"

PY_ARGS=${@:1}

OUT_DIR="./out/"
LOG_DIR=$OUT_DIR/'lidar_20Hz'
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

python -m sAP3D.nusc_20Hz_lidar_input_pkl \
    --data_path $data_path \
    --data_version $data_version \
    $PY_ARGS | tee -a $LOG_DIR/log.txt
```

**2. 执行命令**：
```bash
bash scripts/nusc_20Hz_lidar_input_pkl.sh
```

*   **产出**：`out/lidar_20Hz/20Hz_lidar_infos_val.pkl`

---

### 步骤 2：CenterPoint 推理 (Inference)

**目标**：利用预训练检测器对所有插值帧进行推理，获取初步的 3D 检测框。

**1. 修改配置文件**：
编辑 `assets/centerpoint_20Hz_lidar_input.py`，确保路径指向您的数据和步骤 1 生成的 pkl。

```python
# 修改 data_root
data_root = '../data/nuscenes/'

# 修改 ann_file (train, val, test 三处均需修改为绝对路径)
# 注意路径指向 UniData 根目录下的 out 文件夹
ann_file = '../out/lidar_20Hz/20Hz_lidar_infos_val.pkl'
```

**2. 执行推理**：
(需切换到 mmdetection3d 能够被调用的环境，或者确保 mmdet3d 已安装在当前环境中)

```bash
python mmdetection3d/tools/test.py \
    assets/centerpoint_20Hz_lidar_input.py \
    assets/ckpts/full_centerpoint.pth \
    --format-only \
    --eval-options jsonfile_prefix=work_dirs/centerpoint_20hz_mini/
```

*   **产出**：`work_dirs/centerpoint_20hz_mini/pts_bbox/results_nusc.json`

---

### 步骤 3：构建时序数据库 (Temporal Database)

**目标**：为推理得到的检测框分配全局唯一的 ID (`instance_token`)，建立跨帧关联。

**1. 准备脚本**：保存为 `scripts/nusc_20Hzlidar_instance-token_generator.sh`

```bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# 1. 修改 data_path
# 注意：这里必须写到 'nuscenes' 这一层，不要写到 'v1.0-mini'
data_path="./data/nuscenes"


# 2. 修改 data_version
# 将默认的 trainval 改为 mini
data_version="v1.0-mini"
# data_version="v1.0-trainval"
PY_ARGS=${@:1}

OUT_DIR="./out/"
LOG_DIR=$OUT_DIR/'lidar_20Hz'
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

python -m sAP3D.nusc_lidar20Hz_instance-token_generator \
    --data_path $data_path \
    --data_version $data_version \
    $PY_ARGS | tee -a $LOG_DIR/instance_token_generator.txt
```

*   **产出**：通常位于 `work_dirs/centerpoint_20hz_mini/pts_bbox/results_nusc_with_instance_token.json`。

---

### 步骤 4：生成最终标注 (Annotation Generation)

**目标**：融合原始真值与检测结果，生成最终的 12Hz 数据集。

**1. 准备脚本**：保存为 `scripts/ann_generator.sh`

```bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# 1. 修改 data_path
# 注意：这里必须写到 'nuscenes' 这一层，不要写到 'v1.0-mini'
data_path="./data/nuscenes"

# 2. 修改 data_version
# 将默认的 trainval 改为 mini
data_version="v1.0-mini"
# data_version="v1.0-trainval"

ann_frequency=$1
PY_ARGS=${@:2}

OUT_DIR="./out/"
LOG_DIR=$OUT_DIR/$input_frequency/'2'_$ann_frequency
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

python -m sAP3D.nusc_annotation_generator \
    --data_path $data_path \
    --data_version $data_version \
    --ann_frequency $ann_frequency \
    $PY_ARGS | tee -a $LOG_DIR/log_generate_ann.txt
```

**2. 执行命令**（推荐使用 **Advanced** 策略）：
```bash
# 参数 "12" 代表生成 12Hz 数据
bash scripts/ann_generator.sh 12 \
   --ann_strategy 'advanced' \
   --lidar_inf_rst_path ./work_dirs/centerpoint_20hz_mini/pts_bbox/results_nusc_with_instance_token.json
```