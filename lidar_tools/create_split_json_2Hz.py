import os
import json
import glob

# 1. 检查数据源
# 这里使用相对路径，和你的运行命令保持一致
occ_source = "../data/eval_results_mini_200_200"

if not os.path.exists(occ_source):
    print(f"Error: 找不到目录: {occ_source}")
    # 尝试用绝对路径兜底检查（仅用于提示）
    abs_path = os.path.abspath(occ_source)
    print(f"   (Absolute path checked: {abs_path})")
    exit(1)

# 2. 扫描文件
files = glob.glob(os.path.join(occ_source, "*.npz"))
files.sort()

if len(files) == 0:
    print(f"Error: 目录 {occ_source} 中没有 .npz 文件！")
    exit(1)

print(f"Found {len(files)} .npz files.")

# 3. 准备输出 (去掉后缀!)
output_dir = "data/split"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "generated_2hz_val.json")

# 关键修改：os.path.splitext(basename)[0] 用于去掉 .npz
# 这样 JSON 里存的就是纯文件名 "0a0d6b..."，代码拼上 ".npz" 后就正常了
json_dict = {
    str(i): os.path.splitext(os.path.basename(f))[0] 
    for i, f in enumerate(files)
}

with open(output_file, 'w') as f:
    json.dump(json_dict, f, indent=4)

print(f"Fixed JSON saved to: {output_file}")