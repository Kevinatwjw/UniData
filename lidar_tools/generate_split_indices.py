"""
Script to generate dataset split indices (Train/Val) using NuScenes Official Splits.
Target Format: {"0": "SceneToken/DataToken", "1": "SceneToken/DataToken", ...}

Refactored with Configuration Class and Argument Parsing for standard research engineering.

Author: UniScene Reproduction Assistant
Date: 2025-12-13
"""

import json
import os
import sys
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from nuscenes.utils import splits

@dataclass
class ProjectConfig:
    """
    Configuration dataclass for the split generation process.
    """
    asap_root: str
    output_dir: str
    version: str
    
    # Constants for filtering
    target_sensor: str = "LIDAR_TOP"
    file_ext: str = ".bin"

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_versions = ['v1.0-mini', 'v1.0-trainval', 'v1.0-test']
        if self.version not in valid_versions:
            raise ValueError(f"Invalid version: {self.version}. Supported: {valid_versions}")
        
        if not os.path.exists(self.asap_root):
            raise FileNotFoundError(f"Input directory not found: {os.path.abspath(self.asap_root)}")


def parse_args() -> ProjectConfig:
    """
    Parse command line arguments and return a ProjectConfig object.
    """
    parser = argparse.ArgumentParser(description="Generate UniScene-style Split Indices JSON.")
    
    parser.add_argument(
        "--asap_root", 
        type=str, 
        default="../data/nuscenes/interp_12Hz_trainval",
        help="Path to the ASAP interpolated NuScenes dataset folder containing json metadata."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="../data/split",
        help="Directory where the output JSON files will be saved."
    )
    parser.add_argument(
        "--version", 
        type=str, 
        default="v1.0-mini",
        choices=['v1.0-mini', 'v1.0-trainval', 'v1.0-test'],
        help="NuScenes dataset version to retrieve official splits."
    )

    args = parser.parse_args()
    
    return ProjectConfig(
        asap_root=args.asap_root,
        output_dir=args.output_dir,
        version=args.version
    )


def load_json(path: str) -> list:
    """Safely load a JSON file."""
    if not os.path.exists(path):
        print(f"[Error] File not found: {os.path.abspath(path)}")
        sys.exit(1)
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[Error] Failed to parse JSON {path}: {e}")
        sys.exit(1)


def get_official_scenes(version: str) -> Tuple[List[str], List[str]]:
    """
    Retrieve official scene splits from nuscenes-devkit.
    """
    train_scenes = []
    val_scenes = []

    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = [] 
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    
    return train_scenes, val_scenes


def build_lookup_tables(root_dir: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Build lookup dictionaries for rapid data association.
    Returns:
        sample_to_scene (dict): Map sample_token -> scene_token
        scene_token_to_name (dict): Map scene_token -> scene_name
    """
    print(f"[Info] Building metadata lookup tables...")
    
    scene_path = os.path.join(root_dir, "scene.json")
    sample_path = os.path.join(root_dir, "sample.json")
    
    scenes = load_json(scene_path)
    samples = load_json(sample_path)
    
    scene_token_to_name = {s['token']: s['name'] for s in scenes}
    
    sample_to_scene = {}
    for s in samples:
        if s.get('scene_token'):
            sample_to_scene[s['token']] = s['scene_token']
            
    return sample_to_scene, scene_token_to_name


def process_sensor_data(cfg: ProjectConfig, 
                       sample_to_scene: Dict[str, str], 
                       scene_token_to_name: Dict[str, str],
                       train_scene_set: set,
                       val_scene_set: set) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Core logic: Iterate through sample_data, associate with scenes, and split.
    """
    sample_data_path = os.path.join(cfg.asap_root, "sample_data.json")
    print(f"[Info] Scanning sensor data from: {sample_data_path}")
    
    sample_data = load_json(sample_data_path)
    
    # Intermediate storage: list of tuples (scene_name, timestamp, composite_key)
    train_items = []
    val_items = []
    
    count_skipped = 0
    
    for record in sample_data:
        filename = record.get('filename', '')
        
        # Filter for specific sensor (LIDAR_TOP)
        if cfg.target_sensor in filename and filename.endswith(cfg.file_ext):
            data_token = record.get('token')
            sample_token = record.get('sample_token')
            timestamp = record.get('timestamp')
            
            # 1. Trace back to Scene
            scene_token = sample_to_scene.get(sample_token)
            if not scene_token:
                count_skipped += 1
                continue
                
            scene_name = scene_token_to_name.get(scene_token)
            if not scene_name:
                count_skipped += 1
                continue
            
            # 2. Construct Official Key: SceneToken/DataToken
            composite_key = f"{scene_token}/{data_token}"
            
            # 3. Assign to Split
            sort_tuple = (scene_name, timestamp, composite_key)
            
            if scene_name in train_scene_set:
                train_items.append(sort_tuple)
            elif scene_name in val_scene_set:
                val_items.append(sort_tuple)
    
    if count_skipped > 0:
        print(f"[Warning] Skipped {count_skipped} records due to missing scene associations.")

    # 4. Deterministic Sorting (Scene Name -> Timestamp)
    print("[Info] Sorting indices for deterministic results...")
    train_items.sort(key=lambda x: (x[0], x[1]))
    val_items.sort(key=lambda x: (x[0], x[1]))
    
    # 5. Convert to final dictionary format {"0": "ID", "1": "ID"}
    train_dict = {str(i): item[2] for i, item in enumerate(train_items)}
    val_dict = {str(i): item[2] for i, item in enumerate(val_items)}
    
    return train_dict, val_dict


def main():
    # 1. Parse and Validate Configuration
    cfg = parse_args()
    print("=" * 60)
    print(f"UniScene Split Generator Configuration:")
    print(f"  - ASAP Root:   {os.path.abspath(cfg.asap_root)}")
    print(f"  - Output Dir:  {os.path.abspath(cfg.output_dir)}")
    print(f"  - Version:     {cfg.version}")
    print("=" * 60)

    # 2. Get Official Split Lists
    train_scenes, val_scenes = get_official_scenes(cfg.version)
    print(f"[Info] Loaded '{cfg.version}' splits: {len(train_scenes)} train scenes, {len(val_scenes)} val scenes.")

    # 3. Build Lookup Tables
    sample_to_scene, scene_token_to_name = build_lookup_tables(cfg.asap_root)

    # 4. Process Data
    train_dict, val_dict = process_sensor_data(
        cfg, 
        sample_to_scene, 
        scene_token_to_name,
        set(train_scenes), 
        set(val_scenes)
    )

    # 5. Save Results
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    train_out_name = "generated_nuScenes_nksr_occ_train.json"
    val_out_name = "generated_nuScenes_nksr_occ_val.json"
    
    train_out_path = os.path.join(cfg.output_dir, train_out_name)
    val_out_path = os.path.join(cfg.output_dir, val_out_name)

    print(f"[Info] Saving to disk...")
    
    with open(train_out_path, 'w') as f:
        json.dump(train_dict, f, indent=4)
        
    with open(val_out_path, 'w') as f:
        json.dump(val_dict, f, indent=4)

    # 6. Final Summary
    print("-" * 60)
    print(f"[Success] Generation Completed.")
    print(f"  Train Samples: {len(train_dict):<8} -> {train_out_name}")
    print(f"  Val Samples:   {len(val_dict):<8} -> {val_out_name}")
    print("-" * 60)


if __name__ == "__main__":
    main()

"""
直接运行（默认使用 Mini 配置）：
python generate_official_split_indices.py

通过命令行修改参数（例如路径不同时）：
python generate_official_split_indices.py \
    --asap_root /path/to/your/interp_12Hz_trainval \
    --output_dir ./my_splits \
    --version v1.0-mini

运行全量数据（未来扩展）：
python generate_official_split_indices.py \
    --asap_root ../data/nuscenes/interp_12Hz_trainval \
    --version v1.0-trainval
"""