import os
import sys
import yaml
import argparse
import logging
import numpy as np
from typing import Dict, Tuple, Any, Optional

# 配置日志记录器
def setup_logging(config: Dict[str, Any]) -> None:
    """
    配置全局日志记录格式与输出。
    """
    log_cfg = config.get('logging', {})
    level = getattr(logging, log_cfg.get('level', 'INFO').upper())
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_cfg.get('save_log', False):
        log_path = log_cfg.get('log_path', 'conversion.log')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )

class VoxelConverter:
    """
    处理稠密体素网格与稀疏坐标列表相互转换的类。
    """

    def __init__(self, config_path: str):
        """
        初始化转换器，加载配置文件。
        
        参数:
            config_path (str): YAML配置文件的路径。
        """
        self.config = self._load_config(config_path)
        setup_logging(self.config)
        self.logger = logging.getLogger(__name__)
        
        # 提取核心参数
        self.params = self.config.get('data_params', {})
        self.grid_size = tuple(self.params.get('grid_size', [200, 200, 16]))
        self.empty_label = self.params.get('empty_label', 17)
        self.npz_key = self.params.get('npz_key', 'semantics')

    def _load_config(self, path: str) -> Dict[str, Any]:
        """
        加载并解析YAML配置文件。
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件未找到: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _save_data(self, data: np.ndarray, path: str, as_npz: bool = False) -> None:
        """
        保存数据到磁盘。
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            if as_npz:
                # 保存为压缩的稠密格式
                np.savez_compressed(path, **{self.npz_key: data})
            else:
                # 保存为稀疏列表
                np.save(path, data)
            self.logger.info(f"数据已成功保存至: {path}")
        except Exception as e:
            self.logger.error(f"保存数据失败: {e}")
            raise

    def dense_to_sparse(self, dense_grid: np.ndarray) -> np.ndarray:
        """
        将稠密网格转换为稀疏坐标列表。
        
        逻辑:
            1. 遍历网格，找到所有不等于 empty_label 的体素。
            2. 记录这些体素的 (x, y, z) 坐标和类别 ID。
        
        参数:
            dense_grid (np.ndarray): 形状为 (H, W, D) 的稠密数组。
            
        返回:
            np.ndarray: 形状为 (N, 4) 的稀疏数组，每行格式为 [x, y, z, label]。
        """
        self.logger.info(f"开始转换: 稠密 ({dense_grid.shape}) -> 稀疏")
        
        # 验证维度
        if dense_grid.shape != self.grid_size:
            self.logger.warning(f"输入网格尺寸 {dense_grid.shape} 与配置 {self.grid_size} 不一致")

        # 获取非空体素的掩码
        mask = dense_grid != self.empty_label
        
        # 获取坐标索引 (N, 3)
        # argwhere 返回满足条件的索引
        coords = np.argwhere(mask)
        
        # 获取对应的语义标签 (N, )
        labels = dense_grid[mask]
        
        # 拼接坐标和标签 -> (N, 4)
        sparse_data = np.hstack((coords, labels[:, None]))
        
        self.logger.info(f"转换完成。有效点数量: {sparse_data.shape[0]}, 压缩率: {1 - sparse_data.shape[0]/dense_grid.size:.2%}")
        return sparse_data

    def sparse_to_dense(self, sparse_data: np.ndarray) -> np.ndarray:
        """
        将稀疏坐标列表还原为稠密网格。
        
        逻辑:
            1. 创建一个全为 empty_label 的空网格。
            2. 读取稀疏列表，将对应坐标填入标签。
        
        参数:
            sparse_data (np.ndarray): 形状为 (N, 4) 的稀疏数组。
            
        返回:
            np.ndarray: 形状为 (H, W, D) 的稠密数组。
        """
        self.logger.info(f"开始转换: 稀疏 ({sparse_data.shape}) -> 稠密 {self.grid_size}")
        
        # 初始化填充了空标签的网格
        dense_grid = np.full(self.grid_size, self.empty_label, dtype=np.uint8)
        
        # 解析数据
        coords = sparse_data[:, :3].astype(int)
        labels = sparse_data[:, 3].astype(int)
        
        # 边界检查 (严谨性检查)
        valid_mask = (
            (coords[:, 0] >= 0) & (coords[:, 0] < self.grid_size[0]) &
            (coords[:, 1] >= 0) & (coords[:, 1] < self.grid_size[1]) &
            (coords[:, 2] >= 0) & (coords[:, 2] < self.grid_size[2])
        )
        
        if not np.all(valid_mask):
            invalid_count = np.sum(~valid_mask)
            self.logger.warning(f"检测到 {invalid_count} 个点超出网格定义范围 {self.grid_size}，将被丢弃。")
            coords = coords[valid_mask]
            labels = labels[valid_mask]

        # 填充网格
        # 使用高级索引 (Advanced Indexing)
        dense_grid[coords[:, 0], coords[:, 1], coords[:, 2]] = labels
        
        self.logger.info("转换完成。")
        return dense_grid

    def run(self):
        """
        执行主流程。
        """
        input_path = self.config['io']['input_path']
        output_path = self.config['io']['output_path']
        mode = self.config['mode']

        if not os.path.exists(input_path):
            self.logger.error(f"输入文件不存在: {input_path}")
            return

        try:
            if mode == "dense_to_sparse":
                # 加载 .npz
                with np.load(input_path) as loader:
                    if self.npz_key in loader:
                        data = loader[self.npz_key]
                    else:
                        # 如果指定的key不存在，尝试读取第一个数组
                        first_key = loader.files[0]
                        self.logger.warning(f"Key '{self.npz_key}' 未找到，使用默认 Key: '{first_key}'")
                        data = loader[first_key]
                
                result = self.dense_to_sparse(data)
                self._save_data(result, output_path, as_npz=False)

            elif mode == "sparse_to_dense":
                # 加载 .npy
                data = np.load(input_path)
                
                if data.ndim != 2 or data.shape[1] != 4:
                    raise ValueError(f"稀疏数据格式错误。期望形状 (N, 4)，实际形状 {data.shape}")
                
                result = self.sparse_to_dense(data)
                self._save_data(result, output_path, as_npz=True)

            else:
                self.logger.error(f"不支持的模式: {mode}")

        except Exception as e:
            self.logger.exception(f"处理过程中发生严重错误: {e}")

def main():
    parser = argparse.ArgumentParser(description="3D体素数据格式转换工具 (Dense <-> Sparse)")
    parser.add_argument('--config', type=str, default='configs/conversion_config.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    converter = VoxelConverter(args.config)
    converter.run()

if __name__ == "__main__":
    main()
"""
python scripts/voxel_converter.py --config configs/conversion_config.yaml
"""
