### `lidar_tools/generate_split_indices.py` ###

---

### 1. 脚本概述 (Overview)
该脚本用于生成符合 **UniScene 官方标准结构** 的数据集划分索引文件（`train.json` 和 `val.json`）。它结合了 **NuScenes 官方的场景划分标准**（通过 `nuscenes-devkit` 获取）和 **ASAP 插值后的元数据**，生成用于 DataLoader 索引的键值对。

### 2. 关键函数功能详解 (Function Breakdown)

*   **`main()`**
    *   **功能**：主流程控制。解析命令行参数，加载配置，协调查找表构建、数据筛选、排序和保存。
    *   **关键逻辑**：
        *   调用 `get_official_scenes` 获取官方定义的 Train/Val 场景列表。
        *   遍历 ASAP 生成的 `sample_data.json`，筛选 `LIDAR_TOP` 数据。
        *   **关联逻辑**：通过 `SampleData -> Sample -> Scene` 的链路，确定每一帧所属的场景。
        *   **排序保证**：为了确保索引（"0", "1", ...）的确定性，先按**场景名称**排序，再按**时间戳**排序。

*   **`process_sensor_data(...)`**
    *   **功能**：核心处理逻辑。迭代传感器数据，生成复合键。
    *   **逻辑**：
        *   读取 `sample_data.json`。
        *   **构造官方键 (Composite Key)**：格式为 `SceneToken/DataToken`。这是 UniScene 特有的 ID 格式，用于区分不同场景下的帧。
        *   根据场景名称将数据分流到 `train_items` 或 `val_items` 列表。

*   **`build_lookup_tables(root_dir)`**
    *   **功能**：构建内存中的哈希映射表，加速查询。
    *   **输出**：
        *   `sample_to_scene`: 快速查询样本属于哪个场景。
        *   `scene_token_to_name`: 快速查询场景 Token 对应的名称（如 "scene-0061"）。

### 3. 输入输出与参数 (I/O & Arguments)

**命令行参数 (Argparse):**
| 参数名 | 类型 | 默认值 | 描述 |
| :--- | :--- | :--- | :--- |
| `--asap_root` | `str` | `../data/.../interp_12Hz_trainval` | **输入目录**。包含 ASAP 插帧后生成的 JSON 元数据的文件夹。 |
| `--output_dir` | `str` | `../data/split` | **输出目录**。保存生成的 JSON 文件。 |
| `--version` | `str` | `v1.0-mini` | **数据集版本**。可选 `v1.0-mini`, `v1.0-trainval` 等，决定了场景划分列表。 |

**输出文件:**
*   `generated_nuScenes_nksr_occ_train.json`: 训练集索引 `{ "0": "SceneToken/DataToken", ... }`
*   `generated_nuScenes_nksr_occ_val.json`: 验证集索引。

---

### `lidar_tools/generate_official_structure_mapping.py` ###

---

### 1. 脚本概述 (Overview)
该脚本用于生成 **全量数据的路径映射表** (`occ2lidar.json`)。它与上面的 Split 脚本配合使用，构成了完整的数据索引系统。其核心作用是将“场景/数据 Token”这一复合键映射到磁盘上真实的 LiDAR 文件路径。

### 2. 关键函数功能详解 (Function Breakdown)

*   **`main()`**
    *   **功能**：加载元数据，建立映射，保存结果。
    *   **关键逻辑**：
        *   加载 `sample.json` 和 `sample_data.json`。
        *   过滤出 `LIDAR_TOP` 通道的数据。
        *   **键构造**：通过 `sample_token` 反查 `scene_token`，构造与 Split 文件一致的 `SceneToken/DataToken` 键。
        *   **值提取**：提取 `filename` 字段（例如 `samples/LIDAR_TOP/xxx.bin`）。

### 3. 输入输出 (Input & Output)

*   **输入**：依赖 `ASAP_ROOT` 下的 `sample.json` 和 `sample_data.json`。
*   **输出**：`generated_official_structure_mapping.json`。
    *   **格式**：`{ "SceneToken/DataToken": "relative/path/to/lidar.bin" }`。

---

### `lidar_tools/convert_12Hznuscenes_info_occ2lidar.py` ###

---

### 1. 脚本概述 (Overview)
该脚本负责将 MMDetection3D 或 ASAP 生成的 `.pkl` 元数据文件转换为 `LiDAR-Gen` 模型所需的特定格式。如果不进行此转换，模型的数据加载器（`Occ2LiDARDataset`）将无法读取元数据中的位姿（Pose）和校准信息。

### 2. 关键函数功能详解 (Function Breakdown)

*   **`main()`**
    *   **功能**：读取原始 Info，重组结构，保存新 Info。
    *   **关键逻辑**：
        1.  **加载原始 Info**：读取 `nuscenes_advanced_12Hz_infos_*.pkl`。
        2.  **重组结构**：将平铺的列表转换为 **按场景分组的列表** (`List[List[Tuple]]`)。
            *   外层列表代表场景。
            *   内层列表代表该场景下的帧序列。
            *   每个元素是一个元组 `(frame_token, frame_info_dict)`。
        3.  **兼容性处理**：如果不是 12Hz 数据（如原始 2Hz），脚本会自动调用 NuScenes SDK 补充 `lidar_top_data_token`，确保格式统一。

### 3. 输入输出 (Input & Output)

*   **输入**：
    *   `--version`: 数据集版本 (e.g., `v1.0-mini`)。
    *   `--split`: 数据划分 (`train` 或 `val`)。
    *   **源文件**：`data/nuscenes_mmdet3d-12Hz/nuscenes_..._infos_*.pkl`。
*   **输出**：
    *   `data/infos/..._converted.pkl`: 转换后的文件，直接被 Dataset 类加载。

---

### `lidar_tools/create_split_json_2Hz.py` (及类似辅助脚本) ###

---

### 1. 脚本概述 (Overview)
这是一组轻量级的辅助脚本，主要用于 **自定义推理流程** 或 **快速测试**。当使用自己生成的 Occupancy（`.npz` 文件）进行推理时，通常没有官方复杂的 Scene 结构，因此需要这些脚本来生成简单的索引。

### 2. 功能简介

*   **`create_split_json_2Hz.py`**
    *   **功能**：扫描生成的 `.npz` 文件夹，生成一个简单的文件列表 JSON。
    *   **关键点**：它会**去掉文件名的后缀**（如 `.npz`），防止 DataLoader 重复添加后缀导致 `FileNotFoundError`。
    *   **输出**：`generated_2hz_val.json` -> `{ "0": "token_a", "1": "token_b" }`。

*   **`create_mapping_from_asap_12Hz.py`**
    *   **功能**：生成 **扁平化** 的映射表（Flat Mapping）。
    *   **特点**：Key 仅仅是 32 位的 Token，不包含 Scene 前缀。
    *   **混合映射策略 (Hybrid Mapping)**：它会将 `Sensor Data Token` 和 `Sample Token` 同时作为 Key 指向同一个文件路径，以确保无论推理时使用哪种 Token 都能找到对应的真值文件。

---

### 3. 运行所需的目录架构 (Directory Structure)

为了让 `lidar_tools` 中的脚本正常工作，建议保持以下文件结构：

```text
Project_Root/
├── lidar_tools/                  <-- [本文件夹]
│   ├── generate_split_indices.py
│   ├── generate_official_structure_mapping.py
│   ├── convert_12Hznuscenes_info_occ2lidar.py
│   └── ...
├── data/
│   ├── nuscenes/
│   │   ├── interp_12Hz_trainval/ <-- [关键输入] ASAP 插帧后的元数据文件夹
│   │   │   ├── sample.json
│   │   │   ├── sample_data.json
│   │   │   └── scene.json
│   │   └── v1.0-mini/            <-- 原始数据
│   ├── split/                    <-- [输出] 生成的 JSON 存放处
│   └── infos/                    <-- [输出] 转换后的 PKL 存放处
```

### `lidar_tools/convert_2Hznuscenes_info_occ2lidar.py` ###

---

### 1. 脚本概述 (Overview)
该脚本是 `convert_12Hznuscenes_info_occ2lidar.py` 的特化版本，专门用于处理 **原始 2Hz 数据集（未插值）** 的元数据转换。它解决了在仅使用官方 NuScenes 数据（如 v1.0-mini 原始版本）进行测试时，因缺少 UniScene 自定义字段（如 `is_key_frame`）而导致的数据加载错误。

### 2. 关键函数功能详解 (Function Breakdown)

*   **`process_frame(nusc, frame)`**
    *   **功能**：对单帧数据进行清洗、补全缺失的元数据字段。
    *   **核心逻辑**：
        *   **数据增强**：调用 NuScenes SDK (`nusc.get`) 查询 `sample`、`scene`、`log` 表，补全 `description`（场景描述）、`location`（地理位置）、`timeofday`（时间段）等信息。
        *   **关键帧标记 (Keyfix)**：尝试从 `sample_data` 表中获取 `is_key_frame` 属性。如果查询失败（例如在某些简化版数据中），则默认设为 `True`（因为 2Hz 列表本身只包含关键帧），作为兜底策略。
        *   **相机内参兼容**：将旧版字段名 `cam_intrinsic` 重命名为 `camera_intrinsics`，以适配最新的 DataLoader。

*   **`extract_visibility(nusc, sample_token)`**
    *   **功能**：提取当前帧中所有标注框的可见性等级（Visibility Token）。
    *   **用途**：某些评估指标需要根据物体的可见程度进行过滤（例如忽略被遮挡严重的物体）。

*   **`main()`**
    *   **功能**：加载原始 `.pkl`，迭代处理每一帧，并将结果重组为按场景分组的列表结构，最后保存为 `_converted.pkl`。

### 3. 输入输出 (Input & Output)

*   **输入**：
    *   `--pkl_path`: 手动指定输入的 `.pkl` 文件路径（通常是官方提供的 `nuscenes_infos_temporal_val_scene.pkl`）。
*   **输出**：
    *   `data/infos/..._converted.pkl`: 转换后的文件，结构为 `List[List[Tuple(token, info_dict)]]`。

---

### `lidar_tools/create_lidar_mapping_2Hz.py` ###

---

### 1. 脚本概述 (Overview)
该脚本用于为 **2Hz（原始关键帧）** 数据生成 LiDAR 文件路径映射表。它直接读取原始数据集的 `sample_data.json`，建立 Token 到 `.bin` 文件路径的索引。

### 2. 关键函数功能详解 (Function Breakdown)

*   **混合映射策略 (Hybrid Mapping)**
    *   **功能**：同时支持两种 Token 查询方式，确保兼容性。
    *   **逻辑**：
        1.  **Identity A (Data Token)**: 将 `sample_data.token`（传感器数据ID）映射到文件路径。这是官方代码的标准查询方式。
        2.  **Identity B (Sample Token)**: 将 `sample_data.sample_token`（关键帧ID）也映射到同一个文件路径。
            *   *背景*：UniScene 生成的 Occupancy 文件 (`.npz`) 往往使用 Sample Token 命名。
            *   *作用*：通过这种“双重映射”，无论代码使用哪种 Token 去查表，都能找到对应的 LiDAR 文件，彻底解决 `KeyError`。

### 3. 输入输出 (Input & Output)

*   **输入**：`data/nuscenes/v1.0-mini/sample_data.json`。
*   **输出**：`data/split/generated_2hz_lidar_mapping.json`。

---

### `lidar_tools/create_mapping_from_asap_12Hz.py` ###

---

### 1. 脚本概述 (Overview)
该脚本是 `create_lidar_mapping_2Hz.py` 的**高频 (12Hz) 版本**。它读取的是 **ASAP 插帧算法生成的元数据**，因此包含了所有插值帧（Sweep）的映射关系。这是运行 12Hz 推理任务时的**必备映射表生成器**。

### 2. 关键函数功能详解 (Function Breakdown)

*   **核心逻辑**：
    *   读取 `interp_12Hz_trainval` 文件夹下的 `sample_data.json`。
    *   **过滤**：仅提取 `LIDAR_TOP` 通道的数据。
    *   **混合映射**：同样采用 `Token` 和 `SampleToken` 双重索引策略，但这里的 `SampleToken` 可能包含了 ASAP 算法为插值帧生成的伪关键帧 ID。

### 3. 输入输出 (Input & Output)

*   **输入**：`data/nuscenes/interp_12Hz_trainval/sample_data.json`（ASAP 的产物）。
*   **输出**：`data/split/generated_12hz_lidar_mapping.json`。
    *   **注意**：此文件是全量的，包含了 2Hz 关键帧和 12Hz 插值帧的所有路径信息。即使只跑 2Hz 任务，也可以安全地使用这个 12Hz 映射表（它是 2Hz 表的超集）。