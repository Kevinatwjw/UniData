import pickle
import os
import argparse
import numpy as np
from nuscenes import NuScenes
from tqdm import tqdm

"""
UniScene 专用 2Hz 数据转换器 (Strict Version)
"""

def extract_visibility(nusc, sample_token):
    """从 sample 的 annotations 中提取 visibility"""
    try:
        sample = nusc.get('sample', sample_token)
        visibilities = []
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            visibilities.append(int(ann['visibility_token']))
        
        if not visibilities:
            return np.array([], dtype=np.int32)
        return np.array(visibilities, dtype=np.int32)
    except KeyError:
        return np.array([], dtype=np.int32)

def process_frame(nusc, frame):
    """
    对单帧数据进行清洗和补全
    """
    token = frame['token']
    
    # --- 尝试从 NuScenes SDK 获取元数据 ---
    try:
        # 1. 获取 Sample 记录
        sample = nusc.get('sample', token)
        
        # 2. 获取 Scene 和 Log 用于补全描述信息
        scene_rec = nusc.get('scene', sample['scene_token'])
        log_rec = nusc.get('log', scene_rec['log_token'])
        
        # 3. [关键修改] 获取 LIDAR_TOP 的 sample_data 记录以提取 is_key_frame
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', lidar_token)
        
        # --- 补全字段 ---
        
        # 补全: Description
        if 'description' not in frame:
            frame['description'] = scene_rec['description']
            
        # 补全: Location
        if 'location' not in frame:
            frame['location'] = log_rec['location']
            
        # 补全: Timeofday
        if 'timeofday' not in frame:
            logfile = log_rec['logfile']
            if '-' in logfile:
                frame['timeofday'] = logfile.split('-', 1)[1]
            else:
                frame['timeofday'] = log_rec['date_captured']

        # 补全: is_key_frame (动态提取!)
        if 'is_key_frame' not in frame:
            frame['is_key_frame'] = sd_rec['is_key_frame'] 

    except Exception as e:
        print(f"Warning: Failed to fetch metadata for token {token}: {e}")
        # 如果查询失败，作为兜底策略才设为 True (因为 2Hz 列表里大概率是关键帧)
        if 'is_key_frame' not in frame:
            frame['is_key_frame'] = True

    # --- 补全 Sweeps (保留原始) ---
    if 'sweeps' not in frame:
        frame['sweeps'] = []

    # --- 补全 Visibility ---
    if 'visibility' not in frame:
        frame['visibility'] = extract_visibility(nusc, token)

    # --- 修正相机内参键名 ---
    if 'cams' in frame:
        for cam_name, cam_info in frame['cams'].items():
            if 'cam_intrinsic' in cam_info:
                cam_info['camera_intrinsics'] = cam_info.pop('cam_intrinsic')

    return frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v1.0-trainval", help="e.g. v1.0-mini")
    parser.add_argument("--dataroot", type=str, default="data/nuscenes", help="NuScenes data root")
    parser.add_argument("--split", type=str, default="val", help="train or val")
    parser.add_argument("--pkl_path", type=str, default=None, help="手动指定输入的 2Hz pkl 路径")
    args = parser.parse_args()

    # 1. 确定输入文件路径
    if args.pkl_path:
        ori_info_path = args.pkl_path
    else:
        if "mini" in args.version:
            ori_info_path = f"data/nuscenes_mini_infos_temporal_{args.split}_scene.pkl"
        else:
            ori_info_path = f"data/nuscenes_infos_temporal_{args.split}_scene.pkl"

    if not os.path.exists(ori_info_path):
        print(f"Error: Input file not found: {ori_info_path}")
        return

    print(f"Processing 2Hz Data: {ori_info_path}")
    print(f"Initializing NuScenes ({args.version}) for metadata enrichment...")
    
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    with open(ori_info_path, "rb") as f:
        data = pickle.load(f)
    
    if 'infos' in data and isinstance(data['infos'], dict):
        scenes_dict = data['infos']
    elif isinstance(data, dict) and 'infos' not in data:
        scenes_dict = data 
    else:
        print("Error: Input PKL structure is not a recognizable 2Hz Scene Dict.")
        return

    new_infos = []
    sorted_scene_names = sorted(scenes_dict.keys())
    
    print(f"Converting {len(sorted_scene_names)} scenes...")
    for scene_name in tqdm(sorted_scene_names):
        frames = scenes_dict[scene_name]
        scene_output = []
        
        for frame in frames:
            # 核心处理
            enriched_frame = process_frame(nusc, frame)
            scene_output.append((enriched_frame['token'], enriched_frame))
            
        if scene_output:
            new_infos.append(scene_output)

    # 保存
    output_dir = "data/infos"
    os.makedirs(output_dir, exist_ok=True)
    
    input_name = os.path.basename(ori_info_path).replace('.pkl', '')
    output_path = os.path.join(output_dir, f"{input_name}_converted.pkl")
    
    with open(output_path, "wb") as f:
        pickle.dump(new_infos, f)
        
    print(f"[Success]Converted info saved to:\n   {output_path}")

if __name__ == "__main__":
    main()

"""
python lidar_tools/convert_2Hznuscenes_info_occ2lidar.py \
    --version v1.0-mini \
    --split val \
    --dataroot data/nuscenes \
    --pkl_path data/nuscenes_mini_infos_temporal_val_scene.pkl

python lidar_tools/convert_2Hznuscenes_info_occ2lidar.py \
    --version v1.0-mini \
    --split train \
    --dataroot data/nuscenes \
    --pkl_path data/nuscenes_mini_infos_temporal_train_scene.pkl
"""