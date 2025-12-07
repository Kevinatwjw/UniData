import mmcv
import numpy as np
import os
from collections import defaultdict

def analyze_timeofday_semantics(file_path):
    """
    分析 timeofday 字段的语义含义：是帧级时间戳还是场景级标识符。
    
    Args:
        file_path (str): pkl 文件路径
    """
    print(f"Loading data from: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        data = mmcv.load(file_path)
        infos = data['infos']
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    total_frames = len(infos)
    print(f"Total frames: {total_frames}")

    # 1. 统计唯一值
    # 将所有帧按 timeofday 分组
    group_by_tod = defaultdict(list)
    
    for frame in infos:
        tod = frame.get('timeofday', 'unknown')
        ts = frame.get('timestamp')
        group_by_tod[tod].append(ts)

    unique_tod_count = len(group_by_tod)
    avg_frames_per_tod = total_frames / unique_tod_count if unique_tod_count > 0 else 0

    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS REPORT")
    print("="*60)
    print(f"Unique 'timeofday' strings found: {unique_tod_count}")
    print(f"Average frames sharing the same 'timeofday': {avg_frames_per_tod:.2f}")

    # 2. 逻辑判定
    # 如果每个 timeofday 对应大量不同的 timestamp，则证明它是场景ID
    print("\n" + "-"*60)
    print("HYPOTHESIS TESTING")
    print("-"*60)

    # 抽取前 3 个组进行具体验证
    sample_count = 0
    for tod, timestamps in group_by_tod.items():
        if sample_count >= 3:
            break
        
        num_samples = len(timestamps)
        timestamps = sorted([float(t) for t in timestamps])
        
        # 计算组内时间跨度
        duration = 0
        if num_samples > 1:
            duration = (timestamps[-1] - timestamps[0]) / 1e6  # 假设 timestamp 是微秒
        
        print(f"Group Key (timeofday): {tod}")
        print(f"  - Frame Count: {num_samples}")
        print(f"  - Timestamp Variation: {duration:.4f} seconds")
        
        if num_samples > 1 and duration > 0.1:
            print("  - Conclusion: [SCENE_ID] (Multiple timestamps map to this single timeofday)")
        elif num_samples == 1:
            print("  - Conclusion: [UNCERTAIN] (Only 1 sample, cannot determine)")
        else:
            print("  - Conclusion: [TIMESTAMP] (1-to-1 mapping)")
        
        print("")
        sample_count += 1

    # 3. 最终结论
    print("="*60)
    print("FINAL CONCLUSION")
    ratio = unique_tod_count / total_frames
    if ratio < 0.05:  # 阈值可调整，通常场景数远少于帧数
        print(f"Ratio (Unique IDs / Total Frames) = {ratio:.4f}")
        print("Result: 'timeofday' acts as a SCENE IDENTIFIER (Log Start Time).")
        print("Action: You CAN use 'timeofday' to group frames into scenes.")
    else:
        print(f"Ratio (Unique IDs / Total Frames) = {ratio:.4f}")
        print("Result: 'timeofday' varies frequently, likely a Timestamp.")

if __name__ == '__main__':
    # 请修改为实际的绝对路径
    pkl_path = 'data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val.pkl'
    analyze_timeofday_semantics(pkl_path)