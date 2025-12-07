import mmcv
import numpy as np
import os
import argparse
from tqdm import tqdm
import copy

def _set_prev_next_indices(frame, current_idx, num_frames):
    """
    工具函数：设置场景内的局部整数索引 (prev/next)。
    """
    # 如果是第一帧，prev = -1；否则指向前一个 (i-1)
    frame['prev'] = -1 if current_idx == 0 else current_idx - 1
    
    # 如果是最后一帧，next = -1；否则指向后一个 (i+1)
    frame['next'] = -1 if current_idx == num_frames - 1 else current_idx + 1

def _set_global_scene_indices(frame, global_start, global_end):
    """
    工具函数：设置当前帧所属场景的全局索引范围 (scene_start/scene_end)。
    这两个字段在同一个场景的所有帧中是常数。
    """
    frame['scene_start'] = global_start
    frame['scene_end'] = global_end

# def _set_temporal_neighbors(frame, current_idx, num_frames):
#     """
#     工具函数：构建时序邻居列表 (nice_neighbor_prev/nice_neighbor_next)。
#     用于 VAD/UniAD 的时序融合模块。
#     """
#     # 前向邻居 (倒序，离自己最近的在前面)
#     # 例如当前是 2，结果为 [1, 0]
#     frame['nice_neighbor_prev'] = [idx for idx in range(current_idx - 1, -1, -1)]
    
#     # 后向邻居
#     # 例如当前是 2，总数是 5，结果为 [3, 4]
#     frame['nice_neighbor_next'] = [idx for idx in range(current_idx + 1, num_frames)]

def generate_vad_fields(scene_frames, global_start_idx):
    """
    核心逻辑封装：专门负责计算和填充 VAD/UniAD 所需的时序字段。
    通过调用子工具函数完成具体任务。
    
    Args:
        scene_frames (list): 当前场景下的所有帧（已按时间排序）。
        global_start_idx (int): 当前场景在全局数据集中的起始索引。
        
    Returns:
        list: 包含新增字段的帧列表。
    """
    num_frames = len(scene_frames)
    # 计算该场景的全局结束索引
    global_end_idx = global_start_idx + num_frames - 1
    
    processed_frames = []
    
    for i, frame in enumerate(scene_frames):
        # 使用深拷贝，隔离作用域，防止修改原始引用
        new_frame = copy.deepcopy(frame)

        # 1. 转换 prev/next 为场景内整数索引
        _set_prev_next_indices(new_frame, i, num_frames)

        # 2. 注入全局索引范围
        _set_global_scene_indices(new_frame, global_start_idx, global_end_idx)
        
        # # 3. 构建时序邻居
        # _set_temporal_neighbors(new_frame, i, num_frames)

        processed_frames.append(new_frame)
        
    return processed_frames

def convert_to_scene_format(src_path, dst_path):
    """
    主流程控制函数：负责文件IO、分组、排序，并调用字段生成函数。
    """
    print(f"正在加载原始文件: {src_path}")
    try:
        data = mmcv.load(src_path)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {src_path}")
        return

    # 兼容处理：获取 info 列表
    if isinstance(data, dict) and 'infos' in data:
        infos = data['infos']
        metadata = data.get('metadata', dict(version='v1.0-mini'))
    elif isinstance(data, list):
        infos = data
        metadata = dict(version='v1.0-mini') 
    else:
        raise TypeError("未知的输入文件格式，期望 dict 或 list")

    print(f"原始总帧数: {len(infos)}")
    print("正在按 Scene Token 分组...")

    # 第一步：按场景分组 (Group by Scene)
    scene_map = {}
    for info in infos:
        s_token = info['scene_token']
        if s_token not in scene_map:
            scene_map[s_token] = []
        scene_map[s_token].append(info)

    print(f"共找到 {len(scene_map)} 个独立场景。正在处理字段...")

    # 第二步：处理每个场景
    converted_scenes = {}
    global_frame_counter = 0 

    for s_token in tqdm(scene_map, desc="Processing Scenes"):
        raw_frames = scene_map[s_token]
        
        # 关键：排序必须在调用处理函数之前完成
        raw_frames.sort(key=lambda x: x['timestamp'])
        
        # 调用封装函数处理字段
        new_frames = generate_vad_fields(raw_frames, global_frame_counter)
        
        # 更新全局计数器
        global_frame_counter += len(new_frames)
        
        converted_scenes[s_token] = new_frames

    # 第三步：保存结果
    final_output = dict(
        infos=converted_scenes,
        metadata=metadata
    )

    print(f"正在保存转换结果到: {dst_path}")
    mmcv.dump(final_output, dst_path)
    print(f"转换完成")

if __name__ == '__main__':
    BASE_DIR = '/home/kevin/桌面/VAD/data/nuscenes_mmdet3d-12Hz'
    
    # # 1. 转换验证集 (Val)
    # print("\n--- 开始转换验证集 (Val) ---")
    # val_in = os.path.join(BASE_DIR, 'nuscenes_mini_infos_temporal_val.pkl')
    # val_out = os.path.join(BASE_DIR, 'nuscenes_mini_infos_temporal_val_scene.pkl')
    # convert_to_scene_format(val_in, val_out)

    # 2. 转换训练集 (Train)
    print("\n--- 开始转换训练集 (Train) ---")
    train_in = os.path.join(BASE_DIR, 'nuscenes_mini_infos_temporal_train.pkl')
    train_out = os.path.join(BASE_DIR, 'nuscenes_mini_infos_temporal_train_scene.pkl')
    convert_to_scene_format(train_in, train_out)