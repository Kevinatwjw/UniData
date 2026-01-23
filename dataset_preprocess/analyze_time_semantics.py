#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 timeofday 字段的语义含义

判断 timeofday 是帧级时间戳还是场景级标识符。

用法示例:
  python dataset_preprocess/analyze_time_semantics.py data/infos.pkl
"""

import argparse
import pickle
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


def analyze_timeofday_semantics(file_path: str) -> None:
    """
    分析 timeofday 字段的语义含义：是帧级时间戳还是场景级标识符。
    
    Args:
        file_path: pkl 文件路径
    """
    print(f"正在加载数据: {file_path}")
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
        return

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"加载文件失败: {e}")
        return
    
    # 智能解包
    infos = None
    if isinstance(data, dict):
        if 'infos' in data:
            infos = data['infos']
            if isinstance(infos, dict):
                # 场景中心格式
                all_frames = []
                for scene_frames in infos.values():
                    all_frames.extend(scene_frames)
                infos = all_frames
        else:
            print("错误: 未找到 'infos' 字段")
            return
    elif isinstance(data, list):
        infos = data
    else:
        print(f"错误: 不支持的数据格式: {type(data)}")
        return

    total_frames = len(infos)
    print(f"总帧数: {total_frames}")

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
    print("统计分析报告")
    print("="*60)
    print(f"唯一 'timeofday' 字符串数量: {unique_tod_count}")
    print(f"平均每个 'timeofday' 对应的帧数: {avg_frames_per_tod:.2f}")

    # 2. 逻辑判定
    # 如果每个 timeofday 对应大量不同的 timestamp，则证明它是场景ID
    print("\n" + "-"*60)
    print("假设检验")
    print("-"*60)

    # 抽取前 3 个组进行具体验证
    sample_count = 0
    for tod, timestamps in group_by_tod.items():
        if sample_count >= 3:
            break
        
        num_samples = len(timestamps)
        try:
            timestamps_float = sorted([float(t) for t in timestamps if t is not None])
        except (ValueError, TypeError):
            timestamps_float = []
        
        # 计算组内时间跨度
        duration = 0
        if num_samples > 1 and len(timestamps_float) > 1:
            # 假设 timestamp 是微秒或秒
            if timestamps_float[-1] > 1e10:
                duration = (timestamps_float[-1] - timestamps_float[0]) / 1e6  # 微秒转秒
            else:
                duration = timestamps_float[-1] - timestamps_float[0]
        
        print(f"组键 (timeofday): {tod}")
        print(f"  - 帧数: {num_samples}")
        print(f"  - 时间戳变化范围: {duration:.4f} 秒")
        
        if num_samples > 1 and duration > 0.1:
            print("  - 结论: [场景ID] (多个时间戳映射到同一个timeofday)")
        elif num_samples == 1:
            print("  - 结论: [不确定] (只有1个样本，无法确定)")
        else:
            print("  - 结论: [时间戳] (1对1映射)")
        
        print("")
        sample_count += 1

    # 3. 最终结论
    print("="*60)
    print("最终结论")
    print("="*60)
    ratio = unique_tod_count / total_frames if total_frames > 0 else 0
    if ratio < 0.05:  # 阈值可调整，通常场景数远少于帧数
        print(f"比率 (唯一ID数 / 总帧数) = {ratio:.4f}")
        print("结果: 'timeofday' 作为场景标识符 (日志开始时间)")
        print("建议: 可以使用 'timeofday' 将帧分组到场景中")
    else:
        print(f"比率 (唯一ID数 / 总帧数) = {ratio:.4f}")
        print("结果: 'timeofday' 变化频繁，可能是时间戳")


def main():
    parser = argparse.ArgumentParser(
        description="分析 timeofday 字段的语义含义",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python analyze_time_semantics.py data/infos.pkl
        """
    )
    parser.add_argument('pkl_path', type=str, help='PKL文件路径')
    
    args = parser.parse_args()
    analyze_timeofday_semantics(args.pkl_path)


if __name__ == '__main__':
    main()