import json
import os
import argparse
import sys
from typing import List, Dict, Any
from collections import defaultdict
from tqdm import tqdm

def load_json(path: str) -> Any:
    """
    安全加载 JSON 文件。
    
    Args:
        path: 文件路径。
        
    Returns:
        解析后的 Python 对象 (List 或 Dict)。
    """
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to decode JSON from {path}: {e}")
        sys.exit(1)

def save_json(data: Any, path: str) -> None:
    """
    将数据保存为 JSON 文件。
    """
    try:
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"[INFO] Successfully saved to {path}")
    except Exception as e:
        print(f"[ERROR] Failed to save JSON to {path}: {e}")

def fix_instance_metadata(data_root: str, dataset_folder: str) -> None:
    """
    修复 instance.json 中的元数据不一致问题。
    
    核心逻辑：
    1. 读取 sample.json 构建 [Sample Token -> Timestamp] 的映射。
    2. 读取 sample_annotation.json，按 instance_token 进行分组。
    3. 对每个 Instance 下的 Annotation 列表，依据 Sample 的 Timestamp 进行时序排序。
    4. 重新计算 nbr_annotations，并更新 first_annotation_token 和 last_annotation_token。
    
    Args:
        data_root: NuScenes 数据集根目录。
        dataset_folder: 生成的数据集子文件夹名称。
    """
    target_dir = os.path.join(data_root, dataset_folder)
    sample_path = os.path.join(target_dir, 'sample.json')
    ann_path = os.path.join(target_dir, 'sample_annotation.json')
    instance_path = os.path.join(target_dir, 'instance.json')

    print(f"[INFO] Target Directory: {target_dir}")

    # 1. 加载 sample.json 以获取时间戳基准
    print("[INFO] Loading 'sample.json' for timestamp mapping...")
    samples = load_json(sample_path)
    # 建立 Sample Token -> Timestamp 的哈希表，用于后续快速排序
    sample_time_map = {s['token']: s['timestamp'] for s in samples}

    # 2. 加载 sample_annotation.json (事实数据)
    print("[INFO] Loading 'sample_annotation.json'...")
    annotations = load_json(ann_path)

    # 3. 加载待修复的 instance.json
    print("[INFO] Loading 'instance.json'...")
    instances = load_json(instance_path)

    # 4. 将所有标注按 Instance ID 分组
    print("[INFO] Grouping annotations by instance_token...")
    inst_to_anns = defaultdict(list)
    for ann in tqdm(annotations, desc="Grouping Annotations"):
        inst_token = ann.get('instance_token')
        if inst_token:
            inst_to_anns[inst_token].append(ann)

    # 5. 核心修复循环
    print("[INFO] Updating instance metadata...")
    updated_count = 0
    skipped_count = 0
    warning_count = 0
    
    # 遍历 instance.json 中的每条记录进行校对
    for inst in tqdm(instances, desc="Fixing Instances"):
        inst_token = inst['token']
        related_anns = inst_to_anns.get(inst_token, [])

        # 异常情况处理：如果该实例在 sample_annotation 中没有任何关联数据
        if not related_anns:
            # 这种情况可能发生于数据裁剪或过滤后
            skipped_count += 1
            # 可以选择将计数置零，或保持原样。这里选择更新为事实状态（即0）
            if inst['nbr_annotations'] != 0:
                inst['nbr_annotations'] = 0
                inst['first_annotation_token'] = ""
                inst['last_annotation_token'] = ""
                updated_count += 1
            continue

        # 数据清洗：移除那些指向了不存在 Sample 的无效 Annotation
        valid_anns = []
        for ann in related_anns:
            if ann['sample_token'] in sample_time_map:
                valid_anns.append(ann)
            else:
                warning_count += 1
        
        if not valid_anns:
            skipped_count += 1
            continue

        # 关键步骤：依据 Sample 的时间戳对 Annotation 进行排序
        # 这修正了 ASAP 生成过程中可能存在的乱序问题
        valid_anns.sort(key=lambda x: sample_time_map[x['sample_token']])

        # 获取真实统计数据
        real_count = len(valid_anns)
        real_first_token = valid_anns[0]['token']
        real_last_token = valid_anns[-1]['token']

        # 检查并更新差异
        # 只要有任意一项元数据不符，就进行全量更新
        if (inst['nbr_annotations'] != real_count or
            inst['first_annotation_token'] != real_first_token or
            inst['last_annotation_token'] != real_last_token):
            
            inst['nbr_annotations'] = real_count
            inst['first_annotation_token'] = real_first_token
            inst['last_annotation_token'] = real_last_token
            updated_count += 1

    # 6. 保存修复后的文件
    print(f"[INFO] Saving fixed data to {instance_path}...")
    save_json(instances, instance_path)

    # 7. 输出统计摘要
    print("-" * 60)
    print(f"[SUMMARY] Processing Complete.")
    print(f"Total Instances:         {len(instances)}")
    print(f"Updated Instances:       {updated_count}")
    print(f"Skipped/Empty Instances: {skipped_count}")
    if warning_count > 0:
        print(f"[WARNING] Found {warning_count} annotations with invalid sample tokens (excluded).")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix metadata inconsistencies in instance.json generated by ASAP.")
    
    parser.add_argument('--data_root', type=str, default='./data/nuscenes', 
                        help='Path to the nuscenes root directory.')
    
    # 允许用户指定具体的文件夹名称，增加灵活性
    parser.add_argument('--dataset_folder', type=str, default='advanced_12Hz_trainval',
                        help='Name of the generated dataset folder (e.g., advanced_12Hz_trainval).')
    
    args = parser.parse_args()
    
    fix_instance_metadata(args.data_root, args.dataset_folder)



"""

python ./sAP3D/fill_12Hz__instance_json.py \
    --data_root ./data/nuscenes \
    --dataset_folder advanced_12Hz_mini


python ./sAP3D/fill_12Hz__instance_json.py \
    --data_root ./data/nuscenes \
    --dataset_folder advanced_12Hz_full_mini
"""