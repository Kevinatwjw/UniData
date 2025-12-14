import json
import os

# ================= 配置路径 =================
# 指向 v1.0-mini 数据的绝对路径或相对路径
nusc_mini_path = "./data/nuscenes/v1.0-mini"
json_path = os.path.join(nusc_mini_path, "sample_data.json")

output_json_path = "./data/split/generated_2hz_lidar_mapping.json"
# ===========================================

print(f"Reading: {os.path.abspath(json_path)}")

if not os.path.exists(json_path):
    print("Error: 找不到 sample_data.json")
    exit(1)

with open(json_path, 'r') as f:
    data = json.load(f)

mapping = {}
count_direct = 0
count_sample = 0

print("Generating Hybrid Mapping (Lidar Token + Sample Token)...")

for record in data:
    # 只处理 LIDAR_TOP
    filename = record.get('filename', '')
    if 'LIDAR_TOP' in filename and filename.endswith('.bin'):
        
        filepath = filename
        
        # 1. 映射方式 A: 使用 Sample Data Token (传感器数据 ID)
        token_data = record['token']
        mapping[token_data] = filepath
        count_direct += 1
        
        # 2. 映射方式 B: 使用 Sample Token (关键帧 ID)
        # 只有当它是关键帧(is_key_frame=True)时，sample_token 才能唯一代表这个 lidar 文件
        # 但为了保险，我们把关联的 sample_token 也指向这个文件
        # (注意：如果是 sweep，它也有 sample_token，可能会覆盖 keyframe，
        #  但通常 .npz 是关键帧生成的，所以这里优先保证 Keyframe 正确)
        if record.get('is_key_frame', False):
            token_sample = record['sample_token']
            mapping[token_sample] = filepath
            count_sample += 1

# 保存
os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
with open(output_json_path, "w") as f:
    json.dump(mapping, f, indent=4)

print(f"Success!")
print(f"   - Mapped {count_direct} via Data Token")
print(f"   - Mapped {count_sample} via Sample Token")
print(f"   - Total unique keys: {len(mapping)}")
print(f"Saved to: {output_json_path}")
