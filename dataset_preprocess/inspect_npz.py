#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPY/NPZ文件检查和分析工具

支持：
- 标准NPZ文件（压缩数组）
- 标准NPY文件（单个数组）
- Pickle封装的NPY文件（稀疏数据）
- 可视化功能（2D/3D点云、切片）

用法示例：
  python dataset_preprocess/inspect_npz.py data/labels.npz
  python dataset_preprocess/inspect_npz.py data/points.npy --visualize --slice-z 3
  python dataset_preprocess/inspect_npz.py data/sparse.npy --visualize --view 3d
"""

import numpy as np
import argparse
import os
import pickle
import sys
from typing import Optional, Tuple, Any, Dict, List

# ==========================================
# 1. 全局配置 (Config)
# ==========================================
class InspectConfig:
    """配置类"""
    
    def __init__(self):
        # --- 基础显示设置 ---
        self.show_data = True         # 是否打印具体数据内容（前N个）
        self.show_stats = True        # 是否统计数值分布（最大/最小/均值）
        self.show_unique = True       # 是否统计唯一值（查看分类标签很有用）
        self.show_sparsity = True     # 是否统计稀疏度（非空占比）
        
        # --- 阈值与参数 ---
        self.preview_rows = 10        # 打印数据时显示的行数
        self.max_unique_print = 30    # 唯一值显示数量上限
        self.empty_val = 17           # 视为空/背景的值（用于计算稀疏度，NuScenes通常是0或17）
        
        # --- 智能容错 ---
        self.try_pickle_if_fail = True # 如果 np.load 失败，尝试用 pickle 加载（针对 generate_occ.py）
        
        # --- 可视化设置 ---
        self.enable_visualize = False  # 是否启用可视化
        self.visualize_mode = '2d'    # 可视化模式: '2d', '3d', 'slice'
        self.slice_z = None            # 切片Z值（用于2D可视化）
        self.slice_axis = 2             # 切片轴（0=X, 1=Y, 2=Z）

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
    
    return arr  # 返回数组以便后续可视化

# ==========================================
# 2.5. 可视化功能
# ==========================================

def visualize_array(name: str, arr: np.ndarray, mode: str = '2d', 
                   slice_z: Optional[int] = None, slice_axis: int = 2) -> None:
    """
    可视化数组数据
    
    Args:
        name: 数组名称
        arr: 数组数据
        mode: 可视化模式 ('2d', '3d', 'slice')
        slice_z: 切片值（用于2D可视化）
        slice_axis: 切片轴（0=X, 1=Y, 2=Z）
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib未安装，跳过可视化")
        return
    
    print(f"\n[可视化] 正在生成 {name} 的可视化...")
    
    # 处理点云数据 (N, >=3) 或 (N, >=4)
    if arr.ndim == 2 and arr.shape[1] >= 3:
        coords = arr[:, :3]  # 取前3列作为坐标
        labels = arr[:, -1] if arr.shape[1] >= 4 else None
        
        if mode == '2d' or mode == 'slice':
            # 2D散点图（XY平面）
            plt.figure(figsize=(10, 10))
            if labels is not None:
                # 根据标签着色
                scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, 
                                     s=1, cmap='tab20', alpha=0.6)
                plt.colorbar(scatter, label='Label')
            else:
                plt.scatter(coords[:, 0], coords[:, 1], s=1, alpha=0.6)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'{name} - XY平面投影')
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        elif mode == '3d':
            # 3D散点图
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            if labels is not None:
                scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                                   c=labels, s=1, cmap='tab20', alpha=0.6)
                plt.colorbar(scatter, label='Label')
            else:
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=1, alpha=0.6)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{name} - 3D点云')
            plt.tight_layout()
            plt.show()
    
    # 处理3D体素数据 (H, W, D) 或 (H, W, D, C)
    elif arr.ndim == 3 or (arr.ndim == 4 and arr.shape[3] == 1):
        if arr.ndim == 4:
            arr = arr[:, :, :, 0]  # 取第一个通道
        
        if mode == 'slice' and slice_z is not None:
            # 切片可视化
            if slice_axis == 0:
                slice_data = arr[slice_z, :, :]
                axis_labels = ('Y', 'Z')
            elif slice_axis == 1:
                slice_data = arr[:, slice_z, :]
                axis_labels = ('X', 'Z')
            else:  # slice_axis == 2
                slice_data = arr[:, :, slice_z]
                axis_labels = ('X', 'Y')
            
            plt.figure(figsize=(10, 10))
            plt.imshow(slice_data, cmap='viridis', origin='lower')
            plt.colorbar(label='Value')
            plt.xlabel(axis_labels[0])
            plt.ylabel(axis_labels[1])
            plt.title(f'{name} - 切片 (axis={slice_axis}, index={slice_z})')
            plt.tight_layout()
            plt.show()
        else:
            # 默认显示中间切片
            mid_z = arr.shape[2] // 2
            plt.figure(figsize=(10, 10))
            plt.imshow(arr[:, :, mid_z], cmap='viridis', origin='lower')
            plt.colorbar(label='Value')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'{name} - 中间切片 (Z={mid_z})')
            plt.tight_layout()
            plt.show()
    
    # 处理2D图像数据 (H, W) 或 (H, W, C)
    elif arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] <= 3):
        if arr.ndim == 3:
            if arr.shape[2] == 1:
                arr = arr[:, :, 0]
            elif arr.shape[2] == 3:
                # RGB图像
                plt.figure(figsize=(10, 10))
                plt.imshow(arr, origin='lower')
                plt.title(f'{name} - RGB图像')
                plt.axis('off')
                plt.tight_layout()
                plt.show()
                return
        
        plt.figure(figsize=(10, 10))
        plt.imshow(arr, cmap='viridis', origin='lower')
        plt.colorbar(label='Value')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'{name} - 2D数据')
        plt.tight_layout()
        plt.show()
    
    else:
        print(f"[WARNING] 不支持的数据维度: {arr.shape}，跳过可视化")

# ==========================================
# 3. 智能加载逻辑
# ==========================================

def smart_load_and_inspect(file_path: str) -> None:
    """
    智能加载并分析文件
    
    Args:
        file_path: 文件路径
    """
    if not os.path.exists(file_path):
        print(f"Error: 文件不存在 - {file_path}")
        return

    file_name = os.path.basename(file_path)
    print(f"\n====== 智能分析: {file_name} ======")
    print(f"完整路径: {file_path}")

    raw_data: Any = None
    is_pickle = False
    arrays_to_visualize: List[Tuple[str, np.ndarray]] = []

    # --- 尝试加载 ---
    try:
        if file_path.endswith('.npz'):
            print("识别格式: Standard NPZ (Archive)")
            raw_data = np.load(file_path, allow_pickle=True)
            print(f"包含 Keys: {list(raw_data.files)}")
            print("-" * 50)
            for key in raw_data.files:
                arr = analyze_array(key, raw_data[key])
                if arr is not None and cfg.enable_visualize:
                    arrays_to_visualize.append((key, arr))

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
                    arr = analyze_array("Content", raw_data)
                    if arr is not None and cfg.enable_visualize:
                        arrays_to_visualize.append(("Content", arr))
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
                            arr = analyze_array(k, v, indent=2)
                            if arr is not None and cfg.enable_visualize:
                                arrays_to_visualize.append((k, arr))
                        else:
                            print(f"  Key '{k}': {type(v)}")
                elif isinstance(raw_data, np.ndarray):
                    arr = analyze_array("Pickle_Array", raw_data)
                    if arr is not None and cfg.enable_visualize:
                        arrays_to_visualize.append(("Pickle_Array", arr))
                elif isinstance(raw_data, list):
                    print(f"  数据类型: List (长度 {len(raw_data)})")
                    print(f"  前项示例: {raw_data[0] if len(raw_data)>0 else 'Empty'}")
                    # 如果是点云列表，尝试转换为数组
                    if len(raw_data) > 0 and isinstance(raw_data[0], (list, tuple, np.ndarray)):
                        try:
                            arr = np.array(raw_data)
                            if arr.ndim == 2 and cfg.enable_visualize:
                                arrays_to_visualize.append(("List_Array", arr))
                        except Exception:
                            pass
                else:
                    print(f"  数据类型: {type(raw_data)}")

        else:
            print("未知后缀，尝试作为 Pickle 读取...")
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            print(f"加载成功，数据类型: {type(data)}")
            if isinstance(data, np.ndarray) and cfg.enable_visualize:
                arrays_to_visualize.append(("Pickle_Data", data))

    except Exception as e:
        print(f"\n[FATAL ERROR] 无法读取文件: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 执行可视化
    if cfg.enable_visualize and arrays_to_visualize:
        for name, arr in arrays_to_visualize:
            visualize_array(name, arr, mode=cfg.visualize_mode, 
                          slice_z=cfg.slice_z, slice_axis=cfg.slice_axis)

# ==========================================
# 4. 主入口
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="智能 .npy/.npz/.pkl 文件分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基础分析
  python inspect_npz.py data/labels.npz
  
  # 带可视化（2D点云）
  python inspect_npz.py data/points.npy --visualize
  
  # 3D可视化
  python inspect_npz.py data/points.npy --visualize --view 3d
  
  # 切片可视化
  python inspect_npz.py data/voxels.npz --visualize --slice-z 5
        """
    )
    parser.add_argument("path", type=str, help="文件路径")
    
    # --- 命令行参数覆盖 Config ---
    parser.add_argument("--no_stats", action="store_true", help="关闭数值统计")
    parser.add_argument("--no_unique", action="store_true", help="关闭唯一值统计")
    parser.add_argument("--empty_val", type=int, default=17, help="指定什么值被视为空 (默认17)")
    parser.add_argument("--preview", type=int, default=10, help="预览行数")
    parser.add_argument("--visualize", action="store_true", help="启用可视化")
    parser.add_argument("--view", type=str, choices=['2d', '3d', 'slice'], default='2d',
                       help="可视化模式: 2d=2D投影, 3d=3D点云, slice=切片")
    parser.add_argument("--slice-z", type=int, default=None, 
                       help="切片Z值（用于slice模式）")
    parser.add_argument("--slice-axis", type=int, choices=[0, 1, 2], default=2,
                       help="切片轴: 0=X, 1=Y, 2=Z")

    args = parser.parse_args()

    # 应用命令行参数到 Config
    if args.no_stats:
        cfg.show_stats = False
    if args.no_unique:
        cfg.show_unique = False
    cfg.empty_val = args.empty_val
    cfg.preview_rows = args.preview
    cfg.enable_visualize = args.visualize
    cfg.visualize_mode = args.view
    cfg.slice_z = args.slice_z
    cfg.slice_axis = args.slice_axis

    smart_load_and_inspect(args.path)


"""
分析标准的 Dense NPZ (如 labels.npz):
python ./dataset_preprocess/inspect_npz.py data/results/occupancy/eval_results_mini_200_200_12Hz/val/0a0d6b8c2e884134a3b48df43d54c36a.npz
python ./dataset_preprocess/inspect_npz.py data/gts/scene-0003/1ac0914c98b8488cb3521efeba354496/labels.npz
输出预期：会显示 Shape为(200,200,16)，包含语义标签的分布。
分析 generate_occ.py 生成的 Sparse NPY (实为 Pickle):
python ./dataset_preprocess/inspect_npz.py data/results/video/render_maps/val/0a0d6b8c2e884134a3b48df43d54c36a1/semantic.npz
python ./dataset_preprocess/inspect_npz.py data/results/video/render_maps/val/0a0d6b8c2e884134a3b48df43d54c36a1/depth_data.npz
输出预期：会提示 "识别格式: Pickle Wrapped inside NPY"，然后显示 Shape为(N, 4) 的坐标列表。
"""
