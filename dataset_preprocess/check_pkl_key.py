import pickle
import numpy as np
import sys
from collections import Counter
import pprint

# =======================================================
# 用户配置区域 (CONFIG)
# =======================================================
CONFIG = {
    # 1. 基础文件路径
    'pkl_path': 'data/nuscenes_infos_val_temporal_v3_scene.pkl',
    
    # 2. 功能开关
    'show_first_last_frame': True,  # 是否输出第一帧和最后一帧的原始内容
    'enable_stats': False,            # 是否开启字段统计功能
    'enable_comparison': False,      # 是否开启字段对比功能
    'enable_frame_dump': True,       # 是否开启指定前N帧的详细输出
    
    # 3. 统计功能配置 (当 enable_stats=True 时生效)
    'stats_keys': ['sweeps', 'location', 'timeofday', 'visibility', 'is_key_frame'],
    
    # 4. 对比功能配置 (当 enable_comparison=True 时生效)
    'comparison_pair': ('gt_ego_fut_cmd', 'pose_mode'),
    
    # 5. 详细输出配置 (当 enable_frame_dump=True 时生效)
    'dump_rows': 200,  # 输出前多少帧
    
    # [关键修改] 指定要在 "详细输出" 中查看的字段
    # 写在这里的字段，会显示它的 类型(Type)、形状(Shape) 和 内容(Content)
    'dump_keys': [
        'sweeps'
    ], 
    # 是否开启深度表格化检查，打印每个字段的出现率/类型/shape/示例
    'enable_deep_inspect': True,
    'deep_inspect_top_n_samples': 1,  # 每个字段保存多少个示例值用于展示
}

# =======================================================
# 加载数据
# =======================================================
print(f"[INFO] Loading: {CONFIG['pkl_path']}")
try:
    with open(CONFIG['pkl_path'], 'rb') as f:
        data = pickle.load(f)
except Exception as e:
    print(f"[ERROR] Loading failed: {e}")
    sys.exit(1)

# 智能解包
all_frames = []
if isinstance(data, dict):
    if 'infos' in data:
        content = data['infos']
        if isinstance(content, list): all_frames = content
        elif isinstance(content, dict): 
            for key in content: all_frames.extend(content[key])
    else:
        for key in data:
            if isinstance(data[key], list): all_frames.extend(data[key])
elif isinstance(data, list):
    all_frames = data

print(f"[INFO] Total Frames: {len(all_frames)}")

# ===================== 深度表格化检查 =====================
def deep_inspect(frames, top_n_samples=1):
    """遍历所有帧，统计每个字段的出现率、类型、shape 与示例，并以表格形式输出"""
    from collections import defaultdict

    total = len(frames)
    stats = defaultdict(lambda: {'count': 0, 'types': set(), 'shapes': set(), 'samples': []})

    for frame in frames:
        for k, v in frame.items():
            stat = stats[k]
            stat['count'] += 1
            tname = type(v).__name__
            stat['types'].add(tname)

            # 计算 shape
            shape_str = 'N/A'
            try:
                if hasattr(v, 'shape'):
                    shape_str = str(getattr(v, 'shape'))
                elif isinstance(v, (list, tuple)):
                    # 列表内元素可能是 ndarray 或标量
                    if len(v) == 0:
                        shape_str = '(0,)'
                    else:
                        first = v[0]
                        if hasattr(first, 'shape'):
                            shape_str = f'List[{len(v)}]x{first.shape}'
                        else:
                            import numpy as _np
                            try:
                                shape_str = str(_np.array(v).shape)
                            except Exception:
                                shape_str = f'List(len={len(v)})'
                else:
                    shape_str = 'scalar'
            except Exception:
                shape_str = 'N/A'

            stat['shapes'].add(shape_str)

            # 保存示例值（最多 top_n_samples）
            if len(stat['samples']) < top_n_samples:
                stat['samples'].append(v)

    # 输出表格头
    keys_sorted = sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True)

    print('\n' + '='*120)
    print('Deep Inspect Results: Field | Occurrence% | Types | Shapes | Example')
    print('='*120)
    fmt = '{:<30} {:>10} {:<25} {:<25} {:<40}'
    print(fmt.format('Field', 'Occ%', 'Types', 'Shapes', 'Example (truncated)'))
    print('-'*120)

    for k, s in keys_sorted:
        occ = s['count'] / total * 100 if total > 0 else 0
        types_str = '/'.join(sorted(list(s['types']))[:3])
        shapes_str = ' | '.join(list(s['shapes'])[:2])
        sample_repr = ''
        if len(s['samples']) > 0:
            try:
                sample_repr = repr(s['samples'][0])
            except Exception:
                sample_repr = str(type(s['samples'][0]))
        if len(sample_repr) > 200:
            sample_repr = sample_repr[:197] + '...'

        print(fmt.format(k[:30], f"{occ:6.2f}%", types_str[:25], shapes_str[:25], sample_repr[:40]))

    print('='*120 + '\n')


# 如果配置开启则运行深度检查
if CONFIG.get('enable_deep_inspect', False) and len(all_frames) > 0:
    deep_inspect(all_frames, top_n_samples=CONFIG.get('deep_inspect_top_n_samples', 1))

# =======================================================
# 功能函数定义
# =======================================================

def print_field_stats(frames, field_name):
    """统计字段分布"""
    print("\n" + "="*60)
    print(f"Statistics for Field: {field_name}")
    print("="*60)
    
    counter = Counter()
    missing_count = 0
    
    for frame in frames:
        if field_name not in frame:
            missing_count += 1
            continue
        
        val = frame[field_name]
        
        if isinstance(val, np.ndarray):
            val_key = tuple(val.tolist())
        elif isinstance(val, list):
            val_key = str(val)
        else:
            val_key = val

        try:
            counter[val_key] += 1
        except TypeError:
            counter[str(val_key)] += 1

    if missing_count == len(frames):
        print(f"[WARNING] Field '{field_name}' NOT found in any frame.")
        return

    unique_count = len(counter)
    total_valid = sum(counter.values())
    
    items_to_show = []
    # 如果数据太杂乱，只显示前20个
    if unique_count > 20:
        print(f"[提醒] 数据种类很多 (Unique values: {unique_count})")
        print(f"[INFO] 仅输出前 20 个占比最高的数据:")
        items_to_show = counter.most_common(20)
    else:
        items_to_show = counter.most_common()

    print("-" * 60)
    for val, count in items_to_show:
        percent = (count / total_valid) * 100 if total_valid > 0 else 0
        val_str = str(val)
        if len(val_str) > 50: val_str = val_str[:47] + "..."
        print(f"Value: {val_str:<40} | Count: {count:<6} | {percent:.2f}%")
    
    if unique_count > 20:
        print(f"... (还有 {unique_count - 20} 种数值未显示)")

    if missing_count > 0:
        print(f"\n[INFO] Missing frames: {missing_count}")


def print_frame_detail(frame, idx, target_keys=None, title_prefix="Frame"):
    """输出单帧详细信息：内容、类型、形状"""
    print(f"\n--- {title_prefix} {idx} Detail ---")
    
    # 根据配置决定遍历哪些字段
    keys_to_iter = target_keys if (target_keys and len(target_keys) > 0) else frame.keys()
    
    for key in keys_to_iter:
        if key not in frame:
            print(f"Key: {key:<25} | [WARNING] Field NOT found")
            continue

        val = frame[key]
        val_type = type(val).__name__
        val_shape = "N/A"
        
        if isinstance(val, (np.ndarray, list, tuple)):
            try:
                val_shape = str(np.array(val).shape)
            except:
                val_shape = f"len={len(val)}"
        
        val_str = str(val)
        if len(val_str) > 100:
            val_str = val_str[:100] + " ... [truncated]"
            
        print(f"Key: {key:<25} | Type: {val_type:<10} | Shape: {val_shape:<12} | Content: {val_str}")


# =======================================================
# 主逻辑执行
# =======================================================

# 1. 输出首尾帧 (应用 dump_keys 过滤)
if CONFIG['show_first_last_frame'] and len(all_frames) > 0:
    print("\n" + "#"*80)
    print(">>> 1. First and Last Frame Content (Selected Fields Only)")
    print("#"*80)
    
    # [关键修改] 使用 print_frame_detail 替代 pprint，并传入 filtered keys
    print_frame_detail(all_frames[0], 0, target_keys=CONFIG['dump_keys'], title_prefix="FIRST Frame")
    print_frame_detail(all_frames[-1], len(all_frames)-1, target_keys=CONFIG['dump_keys'], title_prefix="LAST Frame")

# 2. 统计功能 (根据开关)
if CONFIG['enable_stats']:
    print("\n" + "#"*80)
    print(">>> 2. Field Statistics")
    print("#"*80)
    for key in CONFIG['stats_keys']:
        print_field_stats(all_frames, key)

# 3. 对比功能 (根据开关)
if CONFIG['enable_comparison']:
    print("\n" + "#"*80)
    print(">>> 3. Field Comparison")
    print("#"*80)
    field_a, field_b = CONFIG['comparison_pair']
    diff_count = 0
    print(f"Comparing {field_a} vs {field_b}")
    for i, frame in enumerate(all_frames):
        if field_a not in frame or field_b not in frame: continue
        val_a, val_b = frame[field_a], frame[field_b]
        is_same = False
        if isinstance(val_a, np.ndarray) and isinstance(val_b, np.ndarray):
            if val_a.shape == val_b.shape: is_same = np.array_equal(val_a, val_b)
        else:
            is_same = (val_a == val_b)
        if not is_same:
            diff_count += 1
            if diff_count <= 20:
                print(f"Frame {i}: {field_a}={val_a} != {field_b}={val_b}")
    if diff_count == 0: print("[SUCCESS] All values match!")
    else: print(f"[WARNING] Total mismatch frames: {diff_count}")

# 4. 指定前 N 帧详细输出 (根据开关和 dump_keys)
if CONFIG['enable_frame_dump']:
    print("\n" + "#"*80)
    print(f">>> 4. Dump First {CONFIG['dump_rows']} Frames Detail (Selected Fields)")
    print("#"*80)
    
    limit = min(len(all_frames), CONFIG['dump_rows'])
    for i in range(limit):
        print_frame_detail(all_frames[i], i, target_keys=CONFIG['dump_keys'])

print("\n[Done]")