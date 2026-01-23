# 数据集预处理工具集

本目录包含用于检查和预处理数据集的工具。

## 工具列表

### 1. `inspect_pkl.py` - PKL文件检查工具（整合版）

整合了 `check_pkl_key.py`、`check_standard_pkl.py`、`check_12Hz_pkl.py` 的功能。

**功能：**
- 深度结构分析（递归遍历）
- 字段统计和分布分析
- 天气关键词统计
- 字段对比
- PKL键名转换
- 多种PKL格式自动识别

**用法示例：**
```bash
# 深度结构分析
python dataset_preprocess/inspect_pkl.py data/infos.pkl --mode deep --max-depth 5

# 字段统计
python dataset_preprocess/inspect_pkl.py data/infos.pkl --mode stats --keys token timestamp

# 天气分布分析
python dataset_preprocess/inspect_pkl.py data/infos.pkl --mode weather

# 字段对比
python dataset_preprocess/inspect_pkl.py data/infos.pkl --mode compare --field-a gt_ego_fut_cmd --field-b pose_mode

# 键名转换
python dataset_preprocess/inspect_pkl.py data/infos.pkl --mode convert --old-key cam_intrinsic --new-key camera_intrinsics --output fixed.pkl

# 快速摘要
python dataset_preprocess/inspect_pkl.py data/infos.pkl --mode summary --dump-rows 10 --dump-keys token timestamp
```

**参数说明：**
- `--mode`: 检查模式 (`deep`, `stats`, `weather`, `compare`, `convert`, `summary`)
- `--max-depth`: 递归遍历最大深度（deep模式）
- `--keys`: 要统计的字段列表（stats模式）
- `--field-a`, `--field-b`: 对比字段（compare模式）
- `--old-key`, `--new-key`: 键名转换（convert模式）
- `--output`: 输出文件路径（convert模式）
- `--dump-rows`: 输出前N帧详情（summary模式）
- `--dump-keys`: 要输出的字段列表（summary模式）

---

### 2. `inspect_npz.py` - NPY/NPZ文件检查工具（增强版）

整合了 `vis_npy.py` 的可视化功能。

**功能：**
- 标准NPZ文件分析（压缩数组）
- 标准NPY文件分析（单个数组）
- Pickle封装的NPY文件分析（稀疏数据）
- 可视化功能（2D/3D点云、切片）

**用法示例：**
```bash
# 基础分析
python dataset_preprocess/inspect_npz.py data/labels.npz

# 带可视化（2D点云）
python dataset_preprocess/inspect_npz.py data/points.npy --visualize

# 3D可视化
python dataset_preprocess/inspect_npz.py data/points.npy --visualize --view 3d

# 切片可视化
python dataset_preprocess/inspect_npz.py data/voxels.npz --visualize --slice-z 5 --slice-axis 2
```

**参数说明：**
- `--no_stats`: 关闭数值统计
- `--no_unique`: 关闭唯一值统计
- `--empty_val`: 指定什么值被视为空（默认17）
- `--preview`: 预览行数（默认10）
- `--visualize`: 启用可视化
- `--view`: 可视化模式 (`2d`, `3d`, `slice`)
- `--slice-z`: 切片Z值（用于slice模式）
- `--slice-axis`: 切片轴 (`0`=X, `1`=Y, `2`=Z)

---

### 3. `inspect_weights.py` - 权重文件检查工具

查看权重文件内部结构（keys / shapes / dtype / 层级统计 / 前缀聚合）。

**支持格式：**
- PyTorch: `.pt`, `.pth`, `.bin`
- SafeTensors: `.safetensors`
- ONNX: `.onnx`
- TensorFlow Checkpoint: `*.index`

**用法示例：**
```bash
python dataset_preprocess/inspect_weights.py ckpt/model.pth --topk 200
python dataset_preprocess/inspect_weights.py weights.safetensors --summary
python dataset_preprocess/inspect_weights.py model.onnx
python dataset_preprocess/inspect_weights.py ckpt/model.ckpt.index --filter backbone
```

**参数说明：**
- `--topk`: 输出最多N个参数条目（默认200）
- `--filter`: 仅输出包含该字符串的key
- `--summary`: 只输出统计信息（不逐条列出keys）
- `--prefix-level`: 前缀聚合层级（默认2）

---

### 4. `analyze_time_semantics.py` - 时间语义分析工具

分析 `timeofday` 字段的语义含义：判断是帧级时间戳还是场景级标识符。

**用法示例：**
```bash
python dataset_preprocess/analyze_time_semantics.py data/infos.pkl
```

---


## 代码规范

所有工具遵循以下规范：
- 使用类型提示（typing）
- 添加详细的docstring（Google风格）
- 遵循PEP 8规范
- 使用argparse进行命令行参数解析
- 提供清晰的错误信息和使用示例

---

## 依赖项

```bash
# 基础依赖
pip install numpy

# 可视化功能（inspect_npz.py）
pip install matplotlib

# 权重检查（inspect_weights.py）
pip install torch  # PyTorch格式
pip install safetensors  # SafeTensors格式
pip install onnx  # ONNX格式
pip install tensorflow  # TensorFlow格式（可选）
```
