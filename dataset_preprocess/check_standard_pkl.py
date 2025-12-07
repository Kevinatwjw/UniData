# -*- coding: utf-8 -*-
# 文件名: check_standard_pkl.py
# 功能：深度解剖 Occ3D / UniScene / OccWorld 官方 temporal pkl，告诉你到底需要生成哪些字段

import pickle
import numpy as np
from collections import defaultdict
import sys

# ============================== 修改这里 ==============================
PKL_PATH = "data/nuscenes_mmdet3d-12Hz/nuscenes_mini_interp_12Hz_infos_val.pkl"   # ← 改成你的路径
# =====================================================================

print(f"正在加载官方 PKL 文件: {PKL_PATH}")
try:
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)
except Exception as e:
    print(f"加载失败: {e}")
    sys.exit(1)

print(f"Top-level keys: {list(data.keys())}\n")

infos = data.get('infos', None)
if infos is None:
    print("未找到 'infos' 字段！")
    sys.exit(1)

# ====================== Step 1: 判断结构类型 ======================
if isinstance(infos, dict):
    print("数据结构: dict[scene_token] -> list[frame_info]")
    all_frames = []
    for scene_token, frames in infos.items():
        all_frames.extend(frames)
        if len(all_frames) > 1000:  # 防止太大，采样前1000帧即可
            break
else:
    print("数据结构: list[frame_info]")
    all_frames = infos

total_frames = len(all_frames)
print(f"总帧数（用于统计）: {total_frames}\n")

# ====================== Step 2: 深度统计所有 key ======================
key_stats = defaultdict(lambda: {'count': 0, 'types': set(), 'shapes': set(), 'sample': None})

print("正在遍历所有帧，统计字段分布（可能需要几秒）...")
for i, frame in enumerate(all_frames):
    for k, v in frame.items():
        key_stats[k]['count'] += 1
        
        # 类型
        if hasattr(v, '__class__'):
            t = v.__class__.__name__
        else:
            t = type(v).__name__
        key_stats[k]['types'].add(t)
        
        # shape（只对 array/tensor/list）
        if isinstance(v, (np.ndarray, list)):
            if isinstance(v, list) and len(v) > 0:
                if isinstance(v[0], np.ndarray):
                    shape = f"List[{len(v)}]x{v[0].shape}"
                else:
                    shape = f"List[{len(v)}]"
            else:
                shape = (len(v),) if isinstance(v, list) else v.shape
            key_stats[k]['shapes'].add(shape)
        elif hasattr(v, 'shape'):
            key_stats[k]['shapes'].add(v.shape)
        
        # 保存第一个样本用于展示
        if key_stats[k]['sample'] is None:
            key_stats[k]['sample'] = v

    if (i + 1) % 5000 == 0:
        print(f"   已处理 {i+1}/{total_frames} 帧...")

print("\n" + "="*100)
print("官方 PKL 字段深度统计结果（按出现频率排序）")
print("="*100)
print(f"{'Key':<35} {'出现率':<8} {'主要类型':<20} {'Shape 示例'}")
print("-"*100)

sorted_keys = sorted(key_stats.items(), key=lambda x: x[1]['count'], reverse=True)

must_have_keys = []
recommend_keys = []

for key, stat in sorted_keys:
    rate = stat['count'] / total_frames * 100
    types_str = "/".join(sorted(list(stat['types']))[:3])  # 最多显示3种
    shapes_str = " | ".join([str(s) for s in list(stat['shapes'])[:3]])  # 最多显示3种
    
    print(f"{key:<35} {rate:6.2f}%   {types_str:<20} {shapes_str}")

    if rate > 99.9:      # 几乎每帧都有 → 必须生成
        must_have_keys.append(key)
    elif rate > 50:      # 大部分都有 → 强烈推荐
        recommend_keys.append(key)

print("\n" + "="*100)
print("【结论】你生成 PKL 时必须包含的字段（缺失必炸")
print("="*100)
for k in must_have_keys:
    print(f"  - {k}")

print("\n【强烈推荐】包含以下字段（很多模型会用，不加可能报错或效果差）")
for k in recommend_keys:
    if k not in must_have_keys:
        print(f"  - {k}")

# # ====================== Step 3: 重点字段详细展开 ======================
# print("\n\n重点字段详细解析（你最关心的几个）".center(100, "="))

# focus_keys = [
#     'token', 'timestamp', 'scene_token',
#     'lidar2e_r', 'l2e_t', 'lidar2ego_rotation', 'lidar2ego_translation',
#     'e2g_r', 'e2g_t', 'ego2global_rotation', 'ego2global_translation',
#     'gt_boxes', 'gt_names', 'gt_ego_fut_trajs',
#     'can_bus', 'location', 'pose_mode'
# ]

# for k in focus_keys:
#     if k in key_stats:
#         stat = key_stats[k]
#         sample = stat['sample']
#         print(f"\n{k}")
#         print(f"   出现率: {stat['count']/total_frames*100:6.2f}%")
#         print(f"   类型: {list(stat['types'])}")
#         if isinstance(sample, np.ndarray):
#             print(f"   shape: {sample.shape}, dtype: {sample.dtype}")
#             if k == 'gt_ego_fut_trajs':
#                 print(f"   → 未来轨迹维度解释: {sample.shape}")
#                 if sample.ndim == 3 and sample.shape[0] == 1:
#                     print("      实际轨迹在 [0,:,:]，你生成时要加 newaxis")
#                 elif sample.ndim == 2:
#                     print("      就是 (T, 2)，直接生成即可")
#         elif isinstance(sample, list):
#             print(f"   list len: {len(sample)}")
#             if len(sample) > 0 and isinstance(sample[0], np.ndarray):
#                 print(f"   元素 shape: {sample[0].shape}")
#         else:
#             print(f"   示例值: {sample}")
#     else:
#         print(f"\n{k} → 不存在！")