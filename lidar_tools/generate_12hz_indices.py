import json
import os
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

from tqdm import tqdm
from nuscenes.utils import splits

# ==============================================================================
#                                 配置管理 (Configuration)
# ==============================================================================

@dataclass
class ProjectConfig:
    """
    项目配置类：管理输入输出路径及数据集版本参数。
    """
    # [输入] 包含 json 元数据的文件夹路径
    SOURCE_META_DIR: str = "data/nuscenes/v1.0-mini_12Hz"
    
    # [输出] 生成的 json 文件保存目录
    OUTPUT_DIR: str = "data/split"
    
    # [参数] 数据集版本 (用于确定 train/val 划分)
    VERSION: str = "v1.0-mini"
    
    # [参数] 目标传感器通道
    SENSOR_CHANNEL: str = "LIDAR_TOP"

# ==============================================================================
#                                 核心逻辑 (Core Logic)
# ==============================================================================

class IndexGenerator:
    def __init__(self, config: ProjectConfig):
        self.cfg = config
        self.meta_path = Path(self.cfg.SOURCE_META_DIR)
        self.out_path = Path(self.cfg.OUTPUT_DIR)
        
        if not self.meta_path.exists():
            raise FileNotFoundError(f"元数据目录不存在: {self.meta_path.absolute()}")
        
        self.out_path.mkdir(parents=True, exist_ok=True)

    def load_json(self, filename: str) -> List[Dict[str, Any]]:
        file_path = self.meta_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"找不到必要的元数据文件: {filename}")
        print(f"[Info] 正在加载: {filename} ...")
        with open(file_path, 'r') as f:
            return json.load(f)

    def get_scene_splits(self) -> Tuple[List[str], List[str]]:
        version = self.cfg.VERSION
        available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
        base_version = version.replace("_12Hz", "")
        
        assert base_version in available_vers, f"不支持的版本: {version}"

        if base_version == 'v1.0-trainval':
            train_scenes = splits.train
            val_scenes = splits.val
        elif base_version == 'v1.0-test':
            train_scenes = splits.test
            val_scenes = []
        elif base_version == 'v1.0-mini':
            train_scenes = splits.mini_train
            val_scenes = splits.mini_val
        else:
            raise ValueError(f"未知的版本配置: {version}")
            
        print(f"[Info] 场景划分 ({base_version}): Train={len(train_scenes)}, Val={len(val_scenes)}")
        return train_scenes, val_scenes

    def run(self):
        # 1. 加载基础元数据
        scenes_data = self.load_json("scene.json")
        samples_data = self.load_json("sample.json")
        sample_data_list = self.load_json("sample_data.json")

        # 2. 构建辅助映射表
        scene_token_map: Dict[str, str] = {s['token']: s['name'] for s in scenes_data}
        # Sample Token -> Scene Token (仅用于区分 Train/Val 集合)
        sample_to_scene: Dict[str, str] = {s['token']: s['scene_token'] for s in samples_data}

        # 3. 获取场景划分列表
        train_scene_names, val_scene_names = self.get_scene_splits()
        train_scene_set = set(train_scene_names)
        val_scene_set = set(val_scene_names)

        print("[Info] 开始处理传感器数据索引...")
        
        train_indices = []
        val_indices = []
        path_mapping = {}
        
        skipped_count = 0

        for record in tqdm(sample_data_list, desc="Processing Frames"):
            if record['channel'] != self.cfg.SENSOR_CHANNEL:
                continue
                
            lidar_token = record['token']
            sample_token = record['sample_token']
            filename = record['filename']
            timestamp = record['timestamp']

            # 关联场景以确定 Train/Val
            scene_token = sample_to_scene.get(sample_token)
            if not scene_token:
                skipped_count += 1
                continue
                
            scene_name = scene_token_map.get(scene_token)
            if not scene_name:
                continue

            # ================= [关键修正点] =================
            # 必须使用 SampleToken/LidarToken 格式
            # 以匹配 generate_occ.py 生成的物理文件夹结构
            composite_key = f"{sample_token}/{lidar_token}"
            # ===============================================
            
            # 填充映射表 (Key -> 物理 .bin 路径)
            path_mapping[composite_key] = filename

            # 填充数据集列表
            entry = (timestamp, composite_key)
            
            if scene_name in train_scene_set:
                train_indices.append(entry)
            elif scene_name in val_scene_set:
                val_indices.append(entry)

        # 5. 排序与保存
        train_indices.sort(key=lambda x: x[0])
        val_indices.sort(key=lambda x: x[0])

        final_train_dict = {str(i): item[1] for i, item in enumerate(train_indices)}
        final_val_dict = {str(i): item[1] for i, item in enumerate(val_indices)}

        self._save_json("nuScenes_nksr_occ_train_mini.json", final_train_dict)
        self._save_json("nuScenes_nksr_occ_val_mini.json", final_val_dict)
        self._save_json("nuScenes_occ2lidar_nksr_mini.json", path_mapping)

        print("-" * 60)
        print(f"[Success] 处理完成 (为 SampleToken/LidarToken 格式)。")
        print(f"  - Train 样本数: {len(final_train_dict)}")
        print(f"  - Val 样本数:   {len(final_val_dict)}")
        print("-" * 60)

    def _save_json(self, filename: str, data: Any):
        file_path = self.out_path / filename
        print(f"[Info] 保存文件: {file_path}")
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_dir", type=str, default=ProjectConfig.SOURCE_META_DIR)
    parser.add_argument("--output_dir", type=str, default=ProjectConfig.OUTPUT_DIR)
    parser.add_argument("--version", type=str, default=ProjectConfig.VERSION)
    args = parser.parse_args()

    config = ProjectConfig()
    config.SOURCE_META_DIR = args.meta_dir
    config.OUTPUT_DIR = args.output_dir
    config.VERSION = args.version

    IndexGenerator(config).run()


"""
python lidar_tools/generate_12hz_indices.py \
    --meta_dir data/nuscenes/v1.0-mini_12Hz \
    --version v1.0-mini
"""