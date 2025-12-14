import pickle
import argparse
import numpy as np
import sys
from typing import Any, Dict, List, Tuple

class Config:
    def __init__(self, args):
        self.path = args.path
        self.max_depth = args.max_depth

def get_obj_info(obj: Any, max_len: int = 300) -> Tuple[str, str, str]:
    """
    提取对象的类型、形状/大小和内容摘要。
    修改版：尽可能显示实际内容，而不是显示 [Container]。
    """
    type_name = type(obj).__name__
    shape_str = "-"
    
    # 默认先转成字符串，获取全部内容
    # replace 换行符，防止表格错乱
    content_str = str(obj).replace('\n', '')

    if isinstance(obj, np.ndarray):
        type_name = "ndarray"
        shape_str = str(obj.shape)
        if obj.size < 20:
            # 小数组：直接显示内容，例如 [1, 2, 3]
            content_str = str(obj.tolist())
        else:
            # 大数组：显示统计信息，更有意义
            if obj.dtype.kind in 'iu f':
                content_str = f"[Min:{obj.min():.2f} Max:{obj.max():.2f} Mean:{obj.mean():.2f}]"
            else:
                content_str = "[Binary/Object Array]"
    
    elif isinstance(obj, (list, tuple)):
        type_name = "list" if isinstance(obj, list) else "tuple"
        shape_str = f"len={len(obj)}"
        # 不再覆盖 content_str，直接使用上面的 str(obj)
        # 这样像 [234.1, 123.4, 1.0] 这种坐标就能打印出来了
    
    elif isinstance(obj, dict):
        type_name = "dict"
        shape_str = f"keys={len(obj)}"
        # 不再覆盖 content_str，直接显示字典内容
        # 如果字典太大，截断逻辑会在最后处理

    elif isinstance(obj, str):
        type_name = "str"
        shape_str = f"len={len(obj)}"
        # 字符串直接显示

    # 统一截断逻辑：如果内容太长（比如几千个元素的列表），才进行截断
    if len(content_str) > max_len:
        content_str = content_str[:max_len] + "..."
    
    return type_name, shape_str, content_str

def recursive_dot_notation(data: Any, parent_path: str, current_depth: int, max_depth: int, fmt: str):
    """
    以点式表示法（key.subkey.[i]）递归打印结构，模拟参考代码的表格输出。
    """
    if current_depth > max_depth:
        return

    type_name, shape_str, content_str = get_obj_info(data)
    
    # 打印当前行
    print(fmt.format(parent_path[:80], type_name[:15], shape_str[:25], content_str[:60]))

    # 递归逻辑
    if isinstance(data, dict):
        # 仅遍历第一个 key 来避免刷屏？不，这里应该遍历所有 key 以展示完整结构
        # 但为了避免字典过大，可以限制显示前 20 个 key
        keys = sorted(list(data.keys()))
        for i, key in enumerate(keys):
            if i >= 20: 
                print(fmt.format(f"{parent_path}....", "...", "...", "Attributes truncated..."))
                break
            
            new_path = f"{parent_path}.{key}" if parent_path else str(key)
            recursive_dot_notation(data[key], new_path, current_depth + 1, max_depth, fmt)
            
    elif isinstance(data, (list, tuple)):
        # 核心逻辑：只递归第一个元素 [0]，并将其显示为 [i]
        if len(data) > 0:
            new_path = f"{parent_path}.[i]"
            recursive_dot_notation(data[0], new_path, current_depth + 1, max_depth, fmt)

def analyze_structure_and_extract_sample(data: Any) -> Tuple[str, Any, str]:
    """
    自动检测结构并提取一个“最小完整单元”（一帧或一个场景）进行深度分析。
    """
    struct_type = "Unknown"
    sample_data = None
    desc = ""

    # 类型 1: 转换后的数据 (List[List[Tuple]]) -> "以场景为中心"
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], list) and len(data[0]) > 0:
            first_item = data[0][0]
            if isinstance(first_item, tuple) and len(first_item) == 2:
                struct_type = "Converted (Scene-Centric Sequence)"
                # 提取整个数据结构作为根，但为了显示 Root.Scene_0，我们需要构造一个包装字典
                # 这样 recursive_dot_notation 就能正确处理第一层
                sample_data = {"Scene_0": data[0]} 
                desc = "Root -> List[Scenes] -> List[Frames]"
                return struct_type, sample_data, desc

    # 类型 2: 标准 NuScenes 数据 (Dict['infos']) -> "扁平列表"
    if isinstance(data, dict) and 'infos' in data and isinstance(data['infos'], list):
        struct_type = "Standard (Flat Frame List)"
        # 同样，构造一个包装字典以便展示层级
        if len(data['infos']) > 0:
            sample_data = {"infos_sample": data['infos'][:1]} # 取包含一帧的列表
        desc = "Root -> Dict['infos'] -> List[Frames]"
        return struct_type, sample_data, desc

    # 类型 3: 通用字典
    if isinstance(data, dict):
        struct_type = "Generic Dictionary"
        sample_data = data
        desc = "Root -> Dict"
        return struct_type, sample_data, desc
    
    return struct_type, data, "Generic"

def print_standard_table(sample_dict: Dict):
    """打印标准字段详情表 (Part 2)"""
    if not isinstance(sample_dict, dict):
        return

    print(f"{'Field / Key':<35} | {'Type':<15} | {'Shape':<20} | {'Content Sample'}")
    print("-" * 120)
    
    # 智能提取最内层的 Info Dict
    target_dict = sample_dict
    
    # Case: Converted -> {'Scene_0': [(token, dict), ...]}
    if 'Scene_0' in sample_dict and isinstance(sample_dict['Scene_0'], list):
         if len(sample_dict['Scene_0']) > 0 and isinstance(sample_dict['Scene_0'][0], tuple):
             target_dict = sample_dict['Scene_0'][0][1]

    # Case: Standard -> {'infos_sample': [dict]}
    elif 'infos_sample' in sample_dict and isinstance(sample_dict['infos_sample'], list):
        if len(sample_dict['infos_sample']) > 0:
            target_dict = sample_dict['infos_sample'][0]

    if isinstance(target_dict, dict):
        for key in sorted(target_dict.keys()):
            val = target_dict[key]
            t, s, c = get_obj_info(val)
            print(f"{key:<35} | {t:<15} | {s:<20} | {c}")
    else:
        print("样本不是标准字典，跳过字段表。")

def main():
    parser = argparse.ArgumentParser(description="PKL 文件深度结构分析器")
    parser.add_argument("path", type=str, help="pkl 文件路径")
    parser.add_argument("--max_depth", type=int, default=10, help="深度遍历的最大层级")
    args = parser.parse_args()
    cfg = Config(args)

    print("=" * 140)
    print(f"Loading: {cfg.path}")
    
    try:
        with open(cfg.path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 1. 自动检测结构
    struct_type, sample_data, hierarchy_desc = analyze_structure_and_extract_sample(data)

    print("-" * 140)
    print(f"Structure Type: {struct_type}")
    print(f"Hierarchy     : {hierarchy_desc}")
    print("-" * 140)

    # 2. 打印标准字段表 (Standard Table) - 用于快速查阅字段
    print("Frame Field Detail (Flattened View):")
    print_standard_table(sample_data)
    print("-" * 140)
    
    # 3. 打印深度点式结构 (Deep Dot-Notation Structure) - 你的重点需求
    print("Deep Hierarchical Structure (Sample Frame/Scene):")
    fmt = '{:<80} {:<15} {:<25} {:<60}'
    print(fmt.format('Path', 'Type', 'Shape', 'Content'))
    print("-" * 180) # 更宽的分隔线以适应内容
    
    # 对提取出的样本数据进行深度递归打印
    # "root" 作为初始路径
    recursive_dot_notation(sample_data, "root", 0, cfg.max_depth, fmt)
    
    print("=" * 140)

if __name__ == "__main__":
    main()
"""
python dataset_preprocess/check_12Hz_pkl.py data/infos/nuscenes_advanced_12Hz_infos_validation_converted.pkl
python dataset_preprocess/check_12Hz_pkl.py data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val.pkl
python dataset_preprocess/check_12Hz_pkl.py data/nuscenes_mmdet3d-12Hz/nuscenes_mini_advanced_12Hz_infos_val.pkl
python dataset_preprocess/check_12Hz_pkl.py data/nuscenes_mini_infos_temporal_val_scene.pkl
python dataset_preprocess/check_12Hz_pkl.py data/infos/nuscenes_mini_infos_temporal_val_scene_converted.pkl
"""