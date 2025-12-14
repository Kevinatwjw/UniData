import pickle

# 你的文件路径
pkl_path = "data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val.pkl"

print(f"Loading {pkl_path} ...")
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

# 1. 检查根目录有哪些 Key
print(f"\nRoot Keys: {list(data.keys())}")

# 2. 重点检查 'scene_tokens' 字段
if 'scene_tokens' in data:
    content = data['scene_tokens']
    print(f"\n[Found 'scene_tokens']")
    print(f"Type: {type(content)}")
    
    if isinstance(content, list):
        print(f"Length: {len(content)}")
        if len(content) > 0:
            first_item = content[0]
            print(f"First Item Type: {type(first_item)}")
            print(f"First Item Content: {first_item}")
            
            # 关键验证：它是不是字符串？
            if isinstance(first_item, str):
                print("\n🔴 破案了！列表里的元素是'字符串' (Scene UUID)。")
                print(f"   代码遍历 '{first_item}' 时，第一个元素就是 '{first_item[0]}' (即 'c')。")
                print("   这就是为什么报错说 'c' is not in list。")
            elif isinstance(first_item, list):
                print("\n🟢 格式正确：列表里的元素是'列表' (Frame Tokens)。")
    else:
        print(f"Content is not a list, it is: {content}")
else:
    print("\n[Missing] 'scene_tokens' key does NOT exist in this file.")
    print("代码如果尝试读取它，应该报 KeyError，而不是 ValueError。")