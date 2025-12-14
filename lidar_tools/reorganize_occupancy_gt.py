import os
import shutil
import argparse
import sys
from pathlib import Path
from typing import Dict, Literal

from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

# ==============================================================================
#                                 配置区域 (Configuration)
# ==============================================================================

class ProjectConfig:
    """
    项目配置类：在此处修改默认路径，或通过命令行参数覆盖。
    """
    # [输入] generate_occ.py 生成的原始 GT 根目录
    # 结构: root / sample_token / lidar_token.npy
    SRC_ROOT: str = "data/GT_occupancy_mini/dense_voxels_with_semantic"

    # [输出] 训练所需的重组后目录
    # 结构: root / scene_token / lidar_token.npy
    DST_ROOT: str = "data/nuscenes_occ"

    # [NuScenes] 数据集版本与路径
    NUSC_VERSION: str = "v1.0-mini"
    NUSC_DATAROOT: str = "data/nuscenes"

    # [操作模式] 推荐使用 'link' 以节省空间
    # 选项: 'link' (软链接), 'copy' (复制), 'move' (移动)
    METHOD: Literal["link", "copy", "move"] = "link"


# ==============================================================================
#                                 核心逻辑 (Core Logic)
# ==============================================================================

def build_sample_to_scene_map(nusc: NuScenes) -> Dict[str, str]:
    """
    构建 SampleToken 到 SceneToken 的映射表。
    用于快速查询，避免在循环中频繁调用 nusc.get()。
    """
    print(f"[Info] 正在构建 Sample -> Scene 映射索引...")
    mapping = {}
    # 遍历所有 sample，直接提取 scene_token
    for sample in nusc.sample:
        mapping[sample['token']] = sample['scene_token']
    return mapping

def process_reorganization(cfg: ProjectConfig):
    """
    执行文件重组的主函数。
    """
    src_path = Path(cfg.SRC_ROOT)
    dst_path = Path(cfg.DST_ROOT)

    # 1. 检查源目录
    if not src_path.exists():
        print(f"[Error] 源目录不存在: {src_path.absolute()}")
        sys.exit(1)

    # 2. 初始化 NuScenes SDK
    print(f"[Info] 初始化 NuScenes ({cfg.NUSC_VERSION})...")
    try:
        nusc = NuScenes(version=cfg.NUSC_VERSION, dataroot=cfg.NUSC_DATAROOT, verbose=False)
    except Exception as e:
        print(f"[Error] NuScenes 初始化失败: {e}")
        sys.exit(1)

    # 3. 构建索引映射
    sample2scene = build_sample_to_scene_map(nusc)

    # 4. 扫描源目录下的 Sample 文件夹
    # 过滤掉非文件夹项
    sample_dirs = [d for d in src_path.iterdir() if d.is_dir()]
    total_samples = len(sample_dirs)
    print(f"[Info] 在源目录中发现 {total_samples} 个 Sample 文件夹。开始重组...")

    processed_files_count = 0
    skipped_count = 0

    # 5. 遍历处理
    for s_dir in tqdm(sample_dirs, desc="Reorganizing"):
        sample_token = s_dir.name

        # 查找对应的 Scene Token
        scene_token = sample2scene.get(sample_token)
        
        if not scene_token:
            # 如果当前 Sample 不在指定的 NuScenes 版本中 (例如用 Mini 跑了 Trainval 的部分数据)
            # print(f"[Warning] Sample {sample_token} 未在当前元数据中找到，跳过。")
            skipped_count += 1
            continue

        # 创建目标 Scene 文件夹
        target_scene_dir = dst_path / scene_token
        target_scene_dir.mkdir(parents=True, exist_ok=True)

        # 处理该文件夹下的所有 .npy 文件
        # 注意：这里 glob 匹配所有 .npy，文件名即 LidarToken，保持不变
        for src_file in s_dir.glob("*.npy"):
            dst_file = target_scene_dir / src_file.name

            if dst_file.exists():
                continue

            try:
                if cfg.METHOD == "link":
                    # 创建软链接 (Symlink) - 推荐
                    # 使用绝对路径以确保链接的稳健性
                    os.symlink(src_file.resolve(), dst_file)
                elif cfg.METHOD == "copy":
                    shutil.copy2(src_file, dst_file)
                elif cfg.METHOD == "move":
                    shutil.move(str(src_file), str(dst_file))
                
                processed_files_count += 1
            except OSError as e:
                print(f"[Error] 操作失败 {src_file.name}: {e}")

    # 6. 输出总结
    print("=" * 60)
    print(f"[Success] 重组完成!")
    print(f"  - 源目录: {src_path.absolute()}")
    print(f"  - 目标目录: {dst_path.absolute()}")
    print(f"  - 处理文件数: {processed_files_count}")
    if skipped_count > 0:
        print(f"  - 跳过 Sample 数: {skipped_count} (不在当前 NuScenes 版本元数据中)")
    print("=" * 60)

# ==============================================================================
#                                 入口 (Entry Point)
# ==============================================================================

if __name__ == "__main__":
    # 支持命令行参数覆盖 Config 类中的默认值
    parser = argparse.ArgumentParser(description="Reorganize Occupancy GT: Sample-based -> Scene-based")
    
    parser.add_argument("--src", type=str, default=ProjectConfig.SRC_ROOT, help="源数据目录 (generate_occ output)")
    parser.add_argument("--dst", type=str, default=ProjectConfig.DST_ROOT, help="目标数据目录 (Output)")
    parser.add_argument("--version", type=str, default=ProjectConfig.NUSC_VERSION, help="NuScenes 版本")
    parser.add_argument("--dataroot", type=str, default=ProjectConfig.NUSC_DATAROOT, help="NuScenes 根目录")
    parser.add_argument("--method", type=str, default=ProjectConfig.METHOD, choices=["link", "copy", "move"], help="文件操作方式")

    args = parser.parse_args()

    # 更新配置
    config = ProjectConfig()
    config.SRC_ROOT = args.src
    config.DST_ROOT = args.dst
    config.NUSC_VERSION = args.version
    config.NUSC_DATAROOT = args.dataroot
    config.METHOD = args.method

    process_reorganization(config)

"""
python lidar_tools/reorganize_occupancy_gt.py \
    --src data/GT_occupancy_mini/dense_voxels_with_semantic \
    --dst data/nuscenes_occ \
    --version v1.0-mini \
    --method link
"""
