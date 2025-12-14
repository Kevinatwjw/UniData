import pickle
from collections import OrderedDict
import os
import argparse
from nuscenes import NuScenes
from tqdm import tqdm

"""
结合 occupancy_gen/dataload_util.py 与 lidar_gen/tools/cfgs/dataset_configs/occ2lidar_dataset_r200*.yaml
可知，下游数据加载器会直接读取一个“按场景顺序组织”的列表结构。本脚本负责把官方
nuScenes 元信息（如 nuscenes_advanced_12Hz_infos_*.pkl）重新整理为该结构，从而为
occupancy/LiDAR 统一生成流程提供一致的索引输入。
"""

if __name__ == "__main__":
    # new_infos 最终会保存一个 List[List[Tuple]]，外层列表对应多个场景，
    # 内层列表中每个元素由 (frame_token, frame_info_dict) 构成，结构与
    # occupancy_gen/dataload_util.py::CustomDataset_* 的读取逻辑完全一致。
    new_infos = []
    # [新增] 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v1.0-trainval", help="v1.0-trainval or v1.0-mini")
    parser.add_argument("--dataroot", type=str, default="data/nuscenes", help="NuScenes data root")
    parser.add_argument("--split", type=str, default="val", help="train or val")
    args = parser.parse_args()

    split = args.split
    
    # [新增] 根据版本自动选择对应的输入文件路径
    # 如果是 mini 版本，文件名通常包含 'mini'
    if args.version == "v1.0-mini":
        # 注意：这里假设你的mini文件名为 nuscenes_mini_interp_12Hz_infos_val.pkl 或类似
        # 如果你的文件名是 advanced，请相应修改下面的文件名
        ori_info_path = f"data/nuscenes_mmdet3d-12Hz/nuscenes_mini_interp_12Hz_infos_{split}.pkl"
    else:
        # 全量版本默认路径
        ori_info_path = f"data/nuscenes_mmdet3d-12Hz/nuscenes_advanced_12Hz_infos_{split}.pkl"

    print(f"Processing version: {args.version}, split: {split}")
    print(f"Loading info file: {ori_info_path}")

    # 读取官方/ASAP 生成的 info 文件，并抽取所有帧的主键 token，用于后续索引。
    info_filename = ori_info_path.split("/")[-1].split(".")[0]
    with open(ori_info_path, "rb") as f:
        infos = pickle.load(f)
    all_frame_tokens = [item["token"] for item in infos["infos"]]
    # 仅当输入并非 12Hz 插值版本时，才需要通过 NuScenes SDK 查询
    # 每帧对应的 LIDAR_TOP sample_data token，并手动构建 scene_tokens。
    # 这样就能兼容 occupancy_gen/12hz_processing/* 中依赖 lidar_top_data_token 的流程。
    if "12Hz" not in ori_info_path:
        print(f"Initializing NuScenes ({args.version})...")
        # [修改] 使用命令行参数传入的版本和路径，不再硬编码
        nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
        for i in range(len(infos["infos"])):
            infos["infos"][i]["lidar_top_data_token"] = nusc.get("sample", infos["infos"][i]["token"])["data"][
                "LIDAR_TOP"
            ]

        all_frame_tokens = [item["lidar_top_data_token"] for item in infos["infos"]]
        scene_tokens = OrderedDict()
        for item in infos["infos"]:
            if item["scene_token"] not in scene_tokens:
                scene_tokens[item["scene_token"]] = []
            # 统一改存 lidar_top token，以便于 occupancy_gen/dataload_util.py::build_clips_new
            # 按照真实的采样频率切分轨迹。
            # scene_tokens[item['scene_token']].append(item['token'])
            # scene_tokens[item['scene_token']].append(item['token'])
            scene_tokens[item["scene_token"]].append(item["lidar_top_data_token"])
        infos["scene_tokens"] = list(scene_tokens.values())
    # 将每个 scene_token 对应的帧，打包为 (frame_token, frame_info) 列表；
    # 这些信息会被 lidar_gen/tools/cfgs/dataset_configs/occ2lidar_dataset_r200_gen.yaml
    # 所引用的 Dataset（occ2lidar_dataset_r200_gen）直接消费，用于生成激光点云。
    for item in tqdm(infos["scene_tokens"]):
        scene_info = []
        # scene_info = {}
        for frame_token in item:
            scene_info.append((frame_token, infos["infos"][all_frame_tokens.index(frame_token)]))
            # scene_info.update({frame_token: infos['infos'][all_frame_tokens.index(frame_token)]})
        new_infos.append(scene_info)
    # 输出文件会被多个模块共享：例如 lidar_gen/tools/cfgs/dataset_configs/occ2lidar_dataset_r200*.yaml
    # 中的 train_info/val_info 默认就指向 *_converted.pkl。
    # 1.在此处指定你想要保存的文件夹路径
    # 例如保存到和输入文件相同的目录: "data/nuscenes_mmdet3d-12Hz"
    output_dir = "data/infos" 
    
    # 自动创建文件夹（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. 拼接完整的输出路径
    output_path = os.path.join(output_dir, f"{info_filename}_converted.pkl")

    print(f"Saving converted info to: {output_path}")
    
    # 3. 保存文件
    with open(output_path, "wb") as f:
        pickle.dump(new_infos, f)


"""
python tools/data_converter/convert_12Hznuscenes_info_occ2lidar.py \
    --version v1.0-mini \
    --split val
python tools/data_converter/convert_12Hznuscenes_info_occ2lidar.py \
    --version v1.0-mini \
    --split train
python tools/data_converter/convert_12Hznuscenes_info_occ2lidar.py \
    --version v1.0-trainval \
    --split val
python tools/data_converter/convert_12Hznuscenes_info_occ2lidar.py \
    --version v1.0-trainval \
    --split train
"""

