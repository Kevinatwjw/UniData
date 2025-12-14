"""
Script to generate LiDAR mapping JSON with Official UniScene Structure.
Target Key Format: "SceneToken/DataToken"
Target Value Format: "Path/To/LiDAR/File.bin"

Author: UniScene Reproduction Assistant
Date: 2025-12-13
"""

import json
import os
import sys
from typing import Dict

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

def build_sample_to_scene_map(samples: list) -> Dict[str, str]:
    """
    Create a lookup dictionary mapping SampleToken to SceneToken.
    """
    mapping = {}
    for record in samples:
        s_token = record.get('token')
        scene_token = record.get('scene_token')
        if s_token and scene_token:
            mapping[s_token] = scene_token
    return mapping

def main():
    # ================= Configuration =================
    # Input: Path to the ASAP interpolated folder containing json files
    # (Based on your provided directory structure)
    asap_root = "../data/nuscenes/interp_12Hz_trainval"
    
    # Output: Path to save the generated mapping file
    output_path = "../data/split/generated_official_structure_mapping.json"
    # ===============================================

    print(f"[Info] Input Directory: {os.path.abspath(asap_root)}")
    
    # 1. Load sample.json to understand Scene context
    sample_json_path = os.path.join(asap_root, "sample.json")
    print(f"[Info] Loading sample index: {sample_json_path}")
    samples = load_json(sample_json_path)
    
    # Build efficient lookup table
    sample_to_scene = build_sample_to_scene_map(samples)
    print(f"[Info] Indexed {len(sample_to_scene)} samples.")

    # 2. Load sample_data.json to get LiDAR files
    data_json_path = os.path.join(asap_root, "sample_data.json")
    print(f"[Info] Loading sensor data: {data_json_path}")
    sample_data = load_json(data_json_path)

    # 3. Generate Mapping
    official_structure_mapping = {}
    count_valid = 0
    count_missing_scene = 0

    print("[Info] Processing records...")
    
    for record in sample_data:
        filename = record.get('filename', '')
        
        # Filter for LIDAR_TOP
        if 'LIDAR_TOP' in filename and filename.endswith('.bin'):
            data_token = record.get('token')
            sample_token = record.get('sample_token')
            
            # Retrieve Scene Token
            scene_token = sample_to_scene.get(sample_token)
            
            if scene_token:
                # Construct Key in Official Format: SceneToken/DataToken
                # Note: If the ASAP process modified tokens to have suffixes (e.g., ...1),
                # data_token will already contain them.
                composite_key = f"{scene_token}/{data_token}"
                
                official_structure_mapping[composite_key] = filename
                count_valid += 1
            else:
                # This happens if a sweep is not associated with a sample in sample.json
                count_missing_scene += 1

    # 4. Save Result
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, 'w') as f:
        json.dump(official_structure_mapping, f, indent=4)

    print("-" * 50)
    print(f"[Success] Generation Complete.")
    print(f"Total Mapped Files: {count_valid}")
    if count_missing_scene > 0:
        print(f"[Warning] Skipped {count_missing_scene} files due to missing scene association.")
    print(f"Output File: {os.path.abspath(output_path)}")
    print("-" * 50)

if __name__ == "__main__":
    main()