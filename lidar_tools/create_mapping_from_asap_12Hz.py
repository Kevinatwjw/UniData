import json
import os
import sys

def main():
    # ================= 配置区域 =================
    # 输入：指向 ASAP 插帧后的文件夹 (包含 sample_data.json)
    asap_output_folder = "../data/nuscenes/interp_12Hz_trainval" 
    
    # 输出：我们把它正确命名为 12Hz，因为它确实包含所有帧的映射
    output_json_path = "../data/split/generated_12hz_lidar_mapping.json"
    # ===========================================

    json_file = os.path.join(asap_output_folder, "sample_data.json")

    print(f"[Info] Reading ASAP Metadata: {os.path.abspath(json_file)}")

    if not os.path.exists(json_file):
        print(f"[Error] 找不到文件: {json_file}")
        sys.exit(1)

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Error] 读取 JSON 失败: {e}")
        sys.exit(1)

    mapping = {}
    count = 0

    print("[Info] Extraction started...")

    for record in data:
        filename = record.get('filename', '')
        token_data = record.get('token', '')
        token_sample = record.get('sample_token', '')

        # 筛选 LIDAR_TOP
        if 'LIDAR_TOP' in filename and filename.endswith('.bin'):
            
            # 1. 映射 Data Token
            mapping[token_data] = filename
            
            # 2. 映射 Sample Token (双重保险)
            if token_sample:
                mapping[token_sample] = filename
            
            count += 1

    # 保存
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(mapping, f, indent=4)

    print(f"[Success] Generated 12Hz Mapping (Count: {count})")
    print(f"          Saved to: {output_json_path}")

if __name__ == "__main__":
    main()