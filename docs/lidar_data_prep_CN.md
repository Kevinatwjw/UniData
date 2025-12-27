## LiDAR 数据预处理工具 (LiDAR Tools)

本模块包含针对 LiDAR 生成模型所需的数据格式转换和索引生成工具。主要用于适配 nuScenes 数据集的原始格式或插值后格式，使其兼容 Occ2LiDAR 的输入要求。

脚本目录：`lidar_tools/`

### 1. 脚本功能详解

#### 1.1 `generate_12hz_indices.py`: 生成数据集分割索引

*   **功能**：扫描 12Hz（或原始）nuScenes 数据集，生成训练集和验证集的样本索引列表 (`.json`)。这是模型训练/推理的第一步，用于划分数据集。
*   **输入**：包含 `scene.json`, `sample.json` 等元数据的文件夹。
*   **输出**：`data/split/` 下的 `nuScenes_nksr_occ_train_mini.json`, `nuScenes_nksr_occ_val_mini.json` 等。

**使用示例：**

```bash
# 生成 v1.0-mini 版本的索引
python lidar_tools/generate_12hz_indices.py \
    --meta_dir data/nuscenes/v1.0-mini_12Hz \
    --version v1.0-mini
```

#### 1.2 `convert_12Hznuscenes_info_occ2lidar.py`: 转换 12Hz 插值数据

*   **功能**：将 12Hz 插值后的 `.pkl` 信息文件转换为 Occ2LiDAR 数据集类所需的列表格式。
*   **适用场景**：已通过 UniData/ASAP 生成了 12Hz 插值数据。
*   **输入**：UniData 生成的 `nuscenes_mini_interp_12Hz_infos_{split}.pkl`。
*   **输出**：`data/infos/` 下的 `*_converted.pkl`。

**使用示例：**

```bash
# 转换验证集
python lidar_tools/convert_12Hznuscenes_info_occ2lidar.py \
    --version v1.0-mini \
    --split val

# 转换训练集
python lidar_tools/convert_12Hznuscenes_info_occ2lidar.py \
    --version v1.0-mini \
    --split train
```

#### 1.3 `convert_2Hznuscenes_info_occ2lidar.py`: 转换 2Hz 原始数据

*   **功能**：将 nuScenes 官方原始的 2Hz `.pkl` 信息文件转换为 Occ2LiDAR 数据集类所需的列表格式。同时会自动补充缺失的元数据（如 `is_key_frame`, `description` 等）。
*   **适用场景**：使用原始 nuScenes 数据进行 Baseline 对比实验。
*   **输入**：官方 `nuscenes_mini_infos_temporal_{split}_scene.pkl`。
*   **输出**：`data/infos/` 下的 `*_converted.pkl`。

**使用示例：**

```bash
# 转换验证集 (需指定 pkl 路径)
python lidar_tools/convert_2Hznuscenes_info_occ2lidar.py \
    --version v1.0-mini \
    --split val \
    --pkl_path data/nuscenes_mini_infos_temporal_val_scene.pkl
```
