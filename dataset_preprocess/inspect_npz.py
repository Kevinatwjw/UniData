import numpy as np
import argparse
import os
import pickle
import sys

# ==========================================
# 1. 全局配置 (Config)
# ==========================================
class InspectConfig:
    def __init__(self):
        # --- 基础显示设置 ---
        self.show_data = True         # 是否打印具体数据内容（前N个）
        self.show_stats = True        # 是否统计数值分布（最大/最小/均值）
        self.show_unique = True       # 是否统计唯一值（查看分类标签很有用）
        self.show_sparsity = True     # 是否统计稀疏度（非空占比）
        
        # --- 阈值与参数 ---
        self.preview_rows = 10        # 打印数据时显示的行数
        self.max_unique_print = 30    # 唯一值显示数量上限 (稍微调大了一点以便显示更多类别)
        self.empty_val = 17           # 视为空/背景的值（用于计算稀疏度，NuScenes通常是0或17）
        
        # --- 智能容错 ---
        self.try_pickle_if_fail = True # 如果 np.load 失败，尝试用 pickle 加载（针对 generate_occ.py）

# 初始化全局配置
cfg = InspectConfig()

# ==========================================
# 2. 核心分析逻辑
# ==========================================

def analyze_array(name, arr, indent=0):
    """分析单个数组并打印信息"""
    prefix = " " * indent
    print(f"{prefix}【Key/Data】: '{name}'")
    
    # 1. 基础信息
    try:
        print(f"{prefix}  - Shape (维度): {arr.shape}")
        print(f"{prefix}  - Dtype (类型): {arr.dtype}")
    except AttributeError:
        # 处理非 numpy 数组（比如 list 或 dict）
        print(f"{prefix}  - Type (对象类型): {type(arr)}")
        if isinstance(arr, list):
            print(f"{prefix}  - Length: {len(arr)}")
            if len(arr) > 0:
                print(f"{prefix}  - Element Type: {type(arr[0])}")
        return

    # 如果不是数值类型（比如字符串数组），跳过后续统计
    if not np.issubdtype(arr.dtype, np.number):
        print(f"{prefix}  - [非数值类型，跳过统计]")
        print("-" * 50)
        return
    # 如果是二维点列表 (N, >=4)，我们把最后一列视为 label 并针对 label 做统计
    stat_arr = arr
    used_label_column = False
    if arr.ndim == 2 and arr.shape[1] >= 4:
        used_label_column = True
        stat_arr = arr[:, -1]
        print(f"{prefix}  - Detected point-list shape: {arr.shape}. 使用最后一列作为 label 进行统计。")

    # 2. 数据预览
    if cfg.show_data:
        print(f"{prefix}  - 数据预览 (前 {cfg.preview_rows} 行):")
        if arr.ndim == 1:
            print(f"{prefix}    {arr[:cfg.preview_rows]}")
        else:
            # 对于高维数组（例如点列表），打印第一维的前几行
            print(f"{prefix}    {arr[:cfg.preview_rows]}")

    # 3. 数值统计（针对 stat_arr）
    if cfg.show_stats:
        try:
            print(f"{prefix}  - 数值范围: Min={np.min(stat_arr)}, Max={np.max(stat_arr)}, Mean={np.mean(stat_arr):.4f}")
        except Exception:
            print(f"{prefix}  - 数值范围: 无法计算（数据类型或结构不支持）")

    # 4. 唯一值统计 (适合语义标签) - 针对 stat_arr
    if cfg.show_unique:
        if np.issubdtype(stat_arr.dtype, np.integer) or stat_arr.size < 1000000:
            unique_vals, counts = np.unique(stat_arr, return_counts=True)
            display_items = []
            limit = min(len(unique_vals), cfg.max_unique_print)
            for i in range(limit):
                display_items.append(f"{unique_vals[i]}: {counts[i]}")
            count_str = ", ".join(display_items)
            if len(unique_vals) > cfg.max_unique_print:
                count_str += " ..."
            print(f"{prefix}  - 包含的唯一值 ({len(unique_vals)}个) [值: 数量]: {count_str}")
        else:
            print(f"{prefix}  - [浮点数/数据量过大，跳过唯一值统计]")

    # 5. 稀疏度/占用率（针对 stat_arr）
    if cfg.show_sparsity:
        total_elements = stat_arr.size
        if total_elements > 0:
            non_empty = np.count_nonzero(stat_arr != cfg.empty_val)
            ratio = non_empty / total_elements * 100
            print(f"{prefix}  - 非空占比 (!= {cfg.empty_val}): {ratio:.2f}% ({non_empty}/{total_elements})")

    print("-" * 50)

# ==========================================
# 3. 智能加载逻辑
# ==========================================

def smart_load_and_inspect(file_path):
    if not os.path.exists(file_path):
        print(f"Error: 文件不存在 - {file_path}")
        return

    file_name = os.path.basename(file_path)
    print(f"\n====== 智能分析: {file_name} ======")
    print(f"完整路径: {file_path}")

    raw_data = None
    is_pickle = False

    # --- 尝试加载 ---
    try:
        if file_path.endswith('.npz'):
            print("识别格式: Standard NPZ (Archive)")
            raw_data = np.load(file_path, allow_pickle=True)
            print(f"包含 Keys: {list(raw_data.files)}")
            print("-" * 50)
            for key in raw_data.files:
                analyze_array(key, raw_data[key])

        elif file_path.endswith('.npy'):
            # 特殊处理：先尝试当做标准 NPY 加载
            try:
                raw_data = np.load(file_path, allow_pickle=True)
                # 检查是否是 pickle 封装的对象数组 (generate_occ.py 的情况)
                if raw_data.dtype == 'O' and not raw_data.shape: 
                    # 这是一个 0-d 对象数组，通常意味着它是直接 pickle dump 的
                    print("识别格式: Pickle Wrapped inside NPY (Pseudo-NPY)")
                    raw_data = raw_data.item() # 提取内部对象
                    is_pickle = True
                else:
                    print("识别格式: Standard NPY (Array)")
                    analyze_array("Content", raw_data)
            except Exception as e:
                if cfg.try_pickle_if_fail:
                    print("Warn: 标准 NPY 加载失败，尝试作为 Pickle 加载...")
                    with open(file_path, 'rb') as f:
                        raw_data = pickle.load(f)
                    is_pickle = True
                else:
                    raise e
            
            # 如果最终被识别为 pickle (即 generate_occ.py 生成的那种稀疏列表)
            if is_pickle:
                print("检测到 Pickle 数据结构:")
                if isinstance(raw_data, dict):
                    print(f"字典 Keys: {list(raw_data.keys())}")
                    for k, v in raw_data.items():
                        if isinstance(v, np.ndarray):
                            analyze_array(k, v, indent=2)
                        else:
                            print(f"  Key '{k}': {type(v)}")
                elif isinstance(raw_data, np.ndarray):
                    analyze_array("Pickle_Array", raw_data)
                elif isinstance(raw_data, list):
                    print(f"  数据类型: List (长度 {len(raw_data)})")
                    print(f"  前项示例: {raw_data[0] if len(raw_data)>0 else 'Empty'}")
                else:
                    print(f"  数据类型: {type(raw_data)}")

        else:
            print("未知后缀，尝试作为 Pickle 读取...")
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            print(f"加载成功，数据类型: {type(data)}")

    except Exception as e:
        print(f"\n[FATAL ERROR] 无法读取文件: {e}")
        import traceback
        traceback.print_exc()

# ==========================================
# 4. 主入口
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="智能 .npy/.npz/.pkl 文件分析工具")
    parser.add_argument("path", type=str, help="文件路径")
    
    # --- 命令行参数覆盖 Config ---
    parser.add_argument("--no_stats", action="store_true", help="关闭数值统计")
    parser.add_argument("--no_unique", action="store_true", help="关闭唯一值统计")
    parser.add_argument("--empty_val", type=int, default=17, help="指定什么值被视为空 (默认17)")
    parser.add_argument("--preview", type=int, default=5, help="预览行数")

    args = parser.parse_args()

    # 应用命令行参数到 Config
    if args.no_stats: cfg.show_stats = False
    if args.no_unique: cfg.show_unique = False
    cfg.empty_val = args.empty_val
    cfg.preview_rows = args.preview

    smart_load_and_inspect(args.path)


"""
分析标准的 Dense NPZ (如 labels.npz):
python ./dataset_preprocess/inspect_npz.py data/results/occupancy/eval_results_mini_800_800_12Hz/0bb62a68055249e381b039bf54b0ccf83.npz
输出预期：会显示 Shape为(200,200,16)，包含语义标签的分布。
分析 generate_occ.py 生成的 Sparse NPY (实为 Pickle):
python ./dataset_preprocess/inspect_npz.py data/GT_occupancy_mini_12Hz_800_800/dense_voxels_with_semantic/0af0feb5b1394b928dd13d648de898f5/c71884fb34d046258c12cf018513d8cc.npy
输出预期：会提示 "识别格式: Pickle Wrapped inside NPY"，然后显示 Shape为(N, 4) 的坐标列表。
"""
