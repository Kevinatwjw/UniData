# UniData Data Processing Guide (I): 12Hz High-Frequency Interpolation via ASAP

**Abstract**: This module serves as the data foundation for the UniData project. Since the original nuScenes dataset only provides keyframe annotations at 2Hz, it cannot meet the training requirements of UniScene for 12Hz continuous video and point cloud generation. This guide will walk you through integrating and running the **ASAP (Autonomous-driving StreAming Perception)** algorithm to generate high-quality 12Hz pseudo-ground-truth annotations.

---

## 0. Project Background and Data Overview

### 0.1 ASAP Framework Analysis

[**ASAP**](https://github.com/JeffWang987/ASAP/tree/main) aims to bridge the huge gap between "sensor sampling frequency" and "manual annotation frequency" in autonomous driving datasets.

In nuScenes, LiDAR data is collected at 20Hz, but official annotations are only provided for keyframes at 2Hz. This means 90% of the data (intermediate frames/Sweeps) remains unsupervised. ASAP offers two strategies to fill this gap:

1.  **Basic Interpolation Strategy (Interp)**: Simple linear interpolation. Fast and suitable for baseline comparisons.
2.  **Advanced Temporal Perception Strategy (Advanced)**: The core contribution. It utilizes a pre-trained detector (CenterPoint) to infer on all frames, constructs temporal associations, corrects interpolation errors, and generates high-quality high-frequency annotations.

### 0.2 Dataset Structure Specification

ASAP strictly relies on the official directory structure of nuScenes. Please ensure your data storage path conforms to the following specification:

```text
UniData/data/nuscenes/
├── maps/               # HD map data
├── samples/            # [Key] Keyframe data, officially annotated (2Hz)
│   ├── CAM_FRONT/
│   ├── LIDAR_TOP/
│   └── ...
├── sweeps/             # [Key] Intermediate frame data, officially unannotated (20Hz)
│   ├── CAM_FRONT/
│   ├── LIDAR_TOP/
│   └── ...
├── v1.0-mini/          # Contains various metadata JSON files
└── v1.0-trainval/      # Use this folder for the full dataset
```

---

## I. Environment and Data Preparation

### 1.1 Environment Dependencies
Please ensure you have completed the `unidata` virtual environment configuration according to the [Environment Setup Guide](../README.md). Core dependencies for this step include:
*   `Python 3.8`
*   `MMDetection3D v0.17.1`
*   `spconv-cu111`

### 1.2 Script Preparation

The ASAP processing workflow relies on several core Shell scripts. Please check the `UniData/scripts/` directory.

> **Note**: The core Python code for ASAP resides in the **`sAP3D/`** folder under the project root directory. The `PYTHONPATH` settings in the scripts below ensure that this module can be correctly invoked.

---

## II. Complete Execution Workflow and Script Details

The ASAP interpolation process is divided into four strictly sequential steps.

> **Note**: The following commands are demonstrated using the **UniData project root directory** as the working path.

### Step 1: Input Generation (20Hz)

**Goal**: Perform linear interpolation on the original data to generate the basic metadata file (`.pkl`) at 20Hz.

**1. Prepare Script**: `scripts/nusc_20Hz_lidar_input_pkl.sh`

```bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# 1. Modify data_path
# Note: This must point to the 'nuscenes' layer, do not write 'v1.0-mini'
data_path="./data/nuscenes"

# 2. Modify data_version
# Change default trainval to mini
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

**2. Execute Command**:
```bash
bash scripts/nusc_20Hz_lidar_input_pkl.sh
```
*   **Output**: `out/lidar_20Hz/20Hz_lidar_infos_val.pkl`

---

### Step 2: CenterPoint Inference

**Goal**: Utilize a pre-trained detector to perform inference on all interpolated frames to obtain preliminary 3D bounding boxes.

**1. Modify Configuration File**:
Edit `assets/centerpoint_20Hz_lidar_input.py` to ensure paths point to your data and the pkl file generated in Step 1.

```python
# Modify data_root
data_root = '../data/nuscenes/'

# Modify ann_file (train, val, test all need to be modified to absolute paths)
# Note: Path points to the out folder under the UniData root directory
ann_file = '../out/lidar_20Hz/20Hz_lidar_infos_val.pkl'
```

**2. Execute Inference**:
(Requires switching to an environment where mmdetection3d can be invoked, or ensuring mmdet3d is installed in the current environment)

```bash
python mmdetection3d/tools/test.py \
    assets/centerpoint_20Hz_lidar_input.py \
    assets/ckpts/full_centerpoint.pth \
    --format-only \
    --eval-options jsonfile_prefix=work_dirs/centerpoint_20hz_mini/
```
*   **Output**: `work_dirs/centerpoint_20hz_mini/pts_bbox/results_nusc.json`

---

### Step 3: Temporal Database Construction

**Goal**: Assign a globally unique ID (`instance_token`) to the detected boxes obtained from inference to establish cross-frame associations.

**1. Prepare Script**: `scripts/nusc_20Hzlidar_instance-token_generator.sh`

```bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# 1. Modify data_path
# Note: This must point to the 'nuscenes' layer, do not write 'v1.0-mini'
data_path="./data/nuscenes"


# 2. Modify data_version
# Change default trainval to mini
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

*   **Output**: Typically located at `work_dirs/centerpoint_20hz_mini/pts_bbox/results_nusc_with_instance_token.json`.

---

### Step 4: Final Annotation Generation

**Goal**: Fuse original ground truth with detection results to generate the final 12Hz dataset.

**1. Prepare Script**: `scripts/ann_generator.sh`

```bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# 1. Modify data_path
# Note: This must point to the 'nuscenes' layer, do not write 'v1.0-mini'
data_path="./data/nuscenes"

# 2. Modify data_version
# Change default trainval to mini
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

**2. Execute Command** (Recommended: **Advanced** strategy):
```bash
# The parameter "12" represents generating 12Hz data
bash scripts/ann_generator.sh 12 \
   --ann_strategy 'advanced' \
   --lidar_inf_rst_path ./work_dirs/centerpoint_20hz_mini/pts_bbox/results_nusc_with_instance_token.json
```