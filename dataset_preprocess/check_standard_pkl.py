# -*- coding: utf-8 -*-
# 文件名: check_standard_pkl.py
# 功能：深度解剖 Occ3D / UniScene / OccWorld 官方 temporal pkl，告诉你到底需要生成哪些字段

import pickle
import numpy as np
from collections import defaultdict
import sys

# ============================== 修改这里 ==============================
PKL_PATH = "data/nuscenes_mini_interp_12Hz_infos_val.pkl"   # ← 改成你的路径
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
# ================ 重构：递归探测原始 PKL 结构并表格化输出 ================

# 配置：为了防止遍历过大对象，可设置遍历上限
MAX_LIST_SAMPLES = 50   # 每个列表最多采样多少个元素进行结构探测
MAX_STR_SAMPLE_LEN = 200

def _get_shape_repr(v):
    try:
        if hasattr(v, 'shape'):
            return str(getattr(v, 'shape'))
        elif isinstance(v, (list, tuple)):
            if len(v) == 0:
                return '(0,)'
            first = v[0]
            if hasattr(first, 'shape'):
                return f'List[{len(v)}]x{first.shape}'
            else:
                try:
                    return str(np.array(v).shape)
                except Exception:
                    return f'List(len={len(v)})'
        else:
            return 'scalar'
    except Exception:
        return 'N/A'


def traverse(obj, path, stats, visited, list_sample_limit=MAX_LIST_SAMPLES, depth=0):
    # 防止循环引用
    obj_id = id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)

    tname = type(obj).__name__
    # 记录当前节点类型
    rec = stats.setdefault(path, {'count': 0, 'types': set(), 'shapes': set(), 'samples': []})
    rec['count'] += 1
    rec['types'].add(tname)

    # 记录 shape/sample for leaf-like objects
    if not isinstance(obj, (dict, list)):
        rec['shapes'].add(_get_shape_repr(obj))
        if len(rec['samples']) < 1:
            try:
                s = repr(obj)
            except Exception:
                s = str(type(obj))
            rec['samples'].append(s[:MAX_STR_SAMPLE_LEN])
        return

    # 对 dict: 遍历键
    if isinstance(obj, dict):
        # 如果 dict 为空，只记录类型信息
        if len(obj) == 0:
            return
        for k, v in obj.items():
            subpath = f"{path}.{k}" if path else str(k)
            traverse(v, subpath, stats, visited, list_sample_limit, depth+1)
        return

    # 对 list: 遍历有限个元素
    if isinstance(obj, list):
        if len(obj) == 0:
            rec['shapes'].add('(0,)')
            return
        rec['shapes'].add(f'List(len={len(obj)})')
        limit = min(len(obj), list_sample_limit)
        for idx in range(limit):
            subpath = f"{path}.[i]" if path else f'[{idx}]'
            traverse(obj[idx], subpath, stats, visited, list_sample_limit, depth+1)
        return


print("开始递归探测 PKL 原始结构（按 pkl 内字段命名路径）...")
stats = {}
visited = set()
traverse(data, '', stats, visited)

# 计算统计量，用于输出出现率
total_encounters = sum(v['count'] for v in stats.values()) if len(stats) > 0 else 1

# 如果存在 infos 且为 dict 且其 values 为 list，视其为 scene->frames，计算 total_frames
frames_total = 0
if 'infos' in data and isinstance(data['infos'], dict):
    try:
        frames_total = sum(len(x) for x in data['infos'].values() if isinstance(x, list))
    except Exception:
        frames_total = 0

# 输出表格
print('\n' + '='*140)
print('PKL 字段深度探测结果（Path | Occ% | Type(s) | Shape(s) | Example）')
print('='*140)
fmt = '{:<60} {:>7} {:<25} {:<20} {:<40}'
print(fmt.format('Path', 'Occ%', 'Types', 'Shapes', 'Example'))
print('-'*140)

for path, info in sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True):
    count = info['count']
    # 优先按 frames_total 计算出现率（如果 path 指向帧内字段）
    if frames_total > 0 and '.[i]' in path:
        occ = count / frames_total * 100
    else:
        occ = count / total_encounters * 100

    types_str = '/'.join(sorted(list(info['types']))[:3])
    shapes_str = ' | '.join(list(info['shapes'])[:2])
    example = info['samples'][0] if len(info['samples']) > 0 else ''
    if len(example) > 200:
        example = example[:197] + '...'

    print(fmt.format(path[:60], f"{occ:6.2f}%", types_str[:25], shapes_str[:20], example[:40]))

print('='*140)

print('\n完成。若需将 scene id 注入到每帧以便兼容旧代码，请告诉我是否在内存中注入（不改文件）。')

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