#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_weights.py
查看权重文件内部结构（keys / shapes / dtype / 层级统计 / 前缀聚合）

支持：
- PyTorch: .pt .pth .bin  (torch.load)
- SafeTensors: .safetensors
- ONNX: .onnx
- TensorFlow Checkpoint: *.index (配套 data 文件)

用法示例：
  python inspect_weights.py /path/to/model.pth --topk 200
  python inspect_weights.py weights.safetensors --summary
  python inspect_weights.py model.onnx
  python inspect_weights.py ckpt/model.ckpt.index

可选参数：
  --topk N        输出最多 N 个参数条目（默认 200）
  --filter STR    仅输出包含 STR 的 key
  --summary       只输出统计信息（不逐条列出 keys）
  --prefix-level L  前缀聚合层级（默认 2，比如 a.b.c -> a.b）
"""

import argparse
import os
import sys
from collections import defaultdict, Counter

def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f}{u}"
        x /= 1024
    return f"{x:.2f}TB"

def dtype_nbytes(dtype) -> int:
    # dtype may be torch dtype, numpy dtype, str, etc.
    s = str(dtype).lower()
    # common map
    if "float64" in s or "double" in s: return 8
    if "float32" in s or "float" in s: return 4
    if "float16" in s or "half" in s: return 2
    if "bfloat16" in s: return 2
    if "int64" in s or "long" in s: return 8
    if "int32" in s or "int" in s: return 4
    if "int16" in s: return 2
    if "int8" in s or "uint8" in s: return 1
    if "bool" in s: return 1
    # fallback
    return 0

def numel_from_shape(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)

def prefix_group(key: str, level: int) -> str:
    parts = key.split(".")
    if level <= 0:
        return key
    return ".".join(parts[: min(level, len(parts))])

def print_table(rows, topk=200):
    # rows: list of dict {key, shape, dtype, numel, bytes}
    # pretty print
    if not rows:
        print("(no parameters found)")
        return
    headers = ["#", "key", "shape", "dtype", "numel", "bytes"]
    # truncate key width
    max_key = min(80, max(len(r["key"]) for r in rows))
    fmt = f"{{:>4}}  {{:<{max_key}}}  {{:<18}}  {{:<12}}  {{:>12}}  {{:>10}}"
    print(fmt.format(*headers))
    print("-" * (4 + 2 + max_key + 2 + 18 + 2 + 12 + 2 + 12 + 2 + 10))
    for i, r in enumerate(rows[:topk], 1):
        key = r["key"]
        if len(key) > max_key:
            key = key[: max_key - 3] + "..."
        print(fmt.format(
            i,
            key,
            str(tuple(r["shape"])),
            str(r["dtype"]),
            f"{r['numel']:,}",
            human_bytes(r["bytes"])
        ))
    if len(rows) > topk:
        print(f"... truncated: showing {topk}/{len(rows)} entries")

def summarize(rows, prefix_level=2):
    total_params = sum(r["numel"] for r in rows)
    total_bytes = sum(r["bytes"] for r in rows)
    dtypes = Counter(str(r["dtype"]) for r in rows)
    prefixes = defaultdict(lambda: {"numel": 0, "bytes": 0, "count": 0})
    for r in rows:
        p = prefix_group(r["key"], prefix_level)
        prefixes[p]["numel"] += r["numel"]
        prefixes[p]["bytes"] += r["bytes"]
        prefixes[p]["count"] += 1

    print("== Summary ==")
    print(f"tensors: {len(rows)}")
    print(f"total params (numel): {total_params:,}")
    print(f"estimated size: {human_bytes(total_bytes)}")
    print("dtypes:")
    for k, v in dtypes.most_common():
        print(f"  {k}: {v}")

    print(f"\n== Prefix aggregation (level={prefix_level}) ==")
    top = sorted(prefixes.items(), key=lambda kv: kv[1]["bytes"], reverse=True)[:30]
    for p, info in top:
        print(f"  {p:<30}  tensors={info['count']:<4}  numel={info['numel']:,}  bytes={human_bytes(info['bytes'])}")

def normalize_state_dict(obj):
    """
    尝试从 torch.load 的结果里提取 state_dict。
    常见格式：
      - 直接就是 state_dict (dict[str, Tensor])
      - {"state_dict": ...}
      - {"model": ...} / {"module": ...} / {"ema": ...}
      - 训练 ckpt: {"model_state_dict": ...} 等
    """
    if not isinstance(obj, dict):
        return None

    # 如果本身就是 "param_name -> tensor"
    # 粗略判断：value 有 shape/size 属性
    def looks_like_tensor(v):
        return hasattr(v, "shape") or hasattr(v, "size")

    if obj and all(isinstance(k, str) for k in obj.keys()) and any(looks_like_tensor(v) for v in obj.values()):
        # 可能是混合 dict，但依然可尝试过滤出 tensor
        tensors = {k: v for k, v in obj.items() if looks_like_tensor(v)}
        if tensors:
            return tensors

    candidate_keys = [
        "state_dict",
        "model",
        "model_state_dict",
        "module",
        "net",
        "network",
        "ema",
        "ema_state_dict",
        "params",
        "weights",
    ]
    for ck in candidate_keys:
        if ck in obj and isinstance(obj[ck], dict):
            sd = normalize_state_dict(obj[ck])
            if sd:
                return sd

    # 还可能嵌套更深：遍历一层
    for _, v in obj.items():
        if isinstance(v, dict):
            sd = normalize_state_dict(v)
            if sd:
                return sd

    return None

def inspect_pytorch(path):
    try:
        import torch
    except Exception as e:
        raise RuntimeError("需要安装 PyTorch 才能读取 .pt/.pth/.bin") from e

    print(f"[PyTorch] loading: {path}")
    obj = torch.load(path, map_location="cpu")
    sd = normalize_state_dict(obj)
    if sd is None:
        raise RuntimeError("无法从该文件中提取 state_dict（可能不是 PyTorch checkpoint，或结构非常规）")

    rows = []
    for k, v in sd.items():
        if not hasattr(v, "shape"):
            continue
        shape = list(v.shape)
        dtype = getattr(v, "dtype", "unknown")
        numel = int(v.numel()) if hasattr(v, "numel") else numel_from_shape(shape)
        nb = dtype_nbytes(dtype)
        est_bytes = numel * nb if nb else 0
        rows.append({"key": k, "shape": shape, "dtype": dtype, "numel": numel, "bytes": est_bytes})
    rows.sort(key=lambda r: r["bytes"], reverse=True)
    return rows

def inspect_safetensors(path):
    try:
        from safetensors import safe_open
    except Exception as e:
        raise RuntimeError("需要安装 safetensors：pip install safetensors") from e

    print(f"[SafeTensors] opening: {path}")
    rows = []
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            shape = list(t.shape)
            dtype = getattr(t, "dtype", "unknown")
            numel = int(t.numel()) if hasattr(t, "numel") else numel_from_shape(shape)
            nb = dtype_nbytes(dtype)
            est_bytes = numel * nb if nb else 0
            rows.append({"key": k, "shape": shape, "dtype": dtype, "numel": numel, "bytes": est_bytes})
    rows.sort(key=lambda r: r["bytes"], reverse=True)
    return rows

def inspect_onnx(path):
    try:
        import onnx
    except Exception as e:
        raise RuntimeError("需要安装 onnx：pip install onnx") from e

    print(f"[ONNX] loading: {path}")
    model = onnx.load(path)
    rows = []

    # initializers are weights
    for init in model.graph.initializer:
        key = init.name
        shape = list(init.dims)
        dtype = init.data_type  # enum int
        # onnx dtype bytes map (partial)
        onnx_bytes = {
            1: 4,  # FLOAT
            2: 1,  # UINT8
            3: 1,  # INT8
            4: 2,  # UINT16
            5: 2,  # INT16
            6: 4,  # INT32
            7: 8,  # INT64
            9: 1,  # BOOL
            10: 2, # FLOAT16
            11: 8, # DOUBLE
            16: 2, # BFLOAT16
        }.get(dtype, 0)
        numel = numel_from_shape(shape) if shape else 0
        est_bytes = numel * onnx_bytes if onnx_bytes else 0
        rows.append({"key": key, "shape": shape, "dtype": f"onnx_dtype_{dtype}", "numel": numel, "bytes": est_bytes})

    rows.sort(key=lambda r: r["bytes"], reverse=True)
    return rows

def inspect_tf_checkpoint(index_path):
    # index_path ends with .index usually
    try:
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError("需要安装 tensorflow 才能读取 TF checkpoint") from e

    print(f"[TensorFlow Checkpoint] loading: {index_path}")
    reader = tf.train.load_checkpoint(index_path.replace(".index", ""))
    var_map = reader.get_variable_to_shape_map()

    rows = []
    for k, shape in var_map.items():
        dtype = reader.get_variable_to_dtype_map().get(k, "unknown")
        numel = numel_from_shape(shape) if shape else 0
        nb = dtype_nbytes(dtype)
        est_bytes = numel * nb if nb else 0
        rows.append({"key": k, "shape": list(shape), "dtype": dtype, "numel": numel, "bytes": est_bytes})

    rows.sort(key=lambda r: r["bytes"], reverse=True)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="权重文件路径（.pt/.pth/.bin/.safetensors/.onnx 或 TF *.index）")
    ap.add_argument("--topk", type=int, default=200, help="最多输出多少条参数（默认 200）")
    ap.add_argument("--filter", type=str, default="", help="仅输出包含该字符串的 key")
    ap.add_argument("--summary", action="store_true", help="只输出统计信息，不逐条列出 keys")
    ap.add_argument("--prefix-level", type=int, default=2, help="前缀聚合层级（默认 2）")
    args = ap.parse_args()

    path = args.path
    if not os.path.exists(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(2)

    lower = path.lower()
    try:
        if lower.endswith(".safetensors"):
            rows = inspect_safetensors(path)
        elif lower.endswith(".onnx"):
            rows = inspect_onnx(path)
        elif lower.endswith(".index"):
            rows = inspect_tf_checkpoint(path)
        elif lower.endswith((".pt", ".pth", ".bin")):
            rows = inspect_pytorch(path)
        else:
            # 尝试按 PyTorch 读一把
            try:
                rows = inspect_pytorch(path)
            except Exception:
                raise RuntimeError("未知扩展名且按 PyTorch 读取失败；请确认文件类型（或改名为正确扩展名）")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.filter:
        rows = [r for r in rows if args.filter in r["key"]]

    summarize(rows, prefix_level=args.prefix_level)
    if not args.summary:
        print()
        print_table(rows, topk=args.topk)

if __name__ == "__main__":
    main()

"""
python ./dataset_preprocess/inspect_weights.py ckpt/GS/weights_MF/depth_net.pth
python ./dataset_preprocess/inspect_weights.py ckpt/GS/weights_MF/gs_net.pth
python ./dataset_preprocess/inspect_weights.py ckpt/GS/weights_MF/pose_net.pth
"""

