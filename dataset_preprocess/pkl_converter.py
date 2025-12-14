import pickle

# 读取你的 mini pkl
src_path = "data/nuscenes_mini_interp_12Hz_infos_train.pkl"
dst_path = "data/nuscenes_mini_interp_12Hz_infos_train_fixed.pkl"

print(f"Loading {src_path}...")
with open(src_path, "rb") as f:
    data = pickle.load(f)

# 遍历所有帧修复 Key
for frame in data['infos']:
    for cam_name, cam_info in frame['cams'].items():
        if 'cam_intrinsic' in cam_info:
            # 将值赋给新 Key
            cam_info['camera_intrinsics'] = cam_info.pop('cam_intrinsic')

print(f"Saving fixed pkl to {dst_path}...")
with open(dst_path, "wb") as f:
    pickle.dump(data, f)

print("Done! Key name mismatch fixed.")