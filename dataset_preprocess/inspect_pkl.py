#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PKL 文件检查和分析工具

整合了 check_pkl_key.py、check_standard_pkl.py、check_12Hz_pkl.py 的功能
支持：
- 深度结构分析（递归遍历）
- 字段统计和分布分析
- 天气关键词统计
- 字段对比
- PKL键名转换
- 多种PKL格式自动识别

用法示例：
  python dataset_preprocess/inspect_pkl.py data/infos.pkl
  python dataset_preprocess/inspect_pkl.py data/infos.pkl --mode deep --max-depth 5
  python dataset_preprocess/inspect_pkl.py data/infos.pkl --mode stats --keys token timestamp
  python dataset_preprocess/inspect_pkl.py data/infos.pkl --convert-key cam_intrinsic camera_intrinsics
"""

import argparse
import pickle
import numpy as np
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple, Optional, Set, Iterable


class PKLInspector:
    """PKL文件检查器"""
    
    def __init__(self, pkl_path: str):
        """
        初始化检查器
        
        Args:
            pkl_path: PKL文件路径
        """
        self.pkl_path = pkl_path
        self.data: Any = None
        self.all_frames: List[Dict] = []
        self._load_data()
    
    def _short_repr(self, obj: Any, max_len: int = 200) -> str:
        """返回对象的简短字符串表示，用于表格展示。"""
        try:
            s = repr(obj)
        except Exception:
            s = str(type(obj))
        s = s.replace("\n", " ")
        if len(s) > max_len:
            s = s[: max_len - 3] + "..."
        return s

    def _value_repr(self, obj: Any, max_len: int = 2000) -> str:
        """返回更详细的字符串表示，用于 tree 模式输出（尽量详细，但允许截断）。"""
        # numpy 数组：优先给出 shape/dtype + 少量内容
        if isinstance(obj, np.ndarray):
            head = f"ndarray(shape={obj.shape}, dtype={obj.dtype})"
            try:
                if obj.size <= 40:
                    detail = repr(obj)
                else:
                    flat = obj.reshape(-1)
                    preview = flat[: min(20, flat.size)]
                    detail = f"preview={preview!r}"
                s = f"{head}; {detail}"
            except Exception:
                s = head
        else:
            try:
                s = repr(obj)
            except Exception:
                s = str(type(obj))
        s = s.replace("\n", " ")
        if len(s) > max_len:
            s = s[: max_len - 3] + "..."
        return s
    
    def _load_data(self) -> None:
        """加载PKL文件并智能解包"""
        print(f"[INFO] 加载文件: {self.pkl_path}")
        try:
            with open(self.pkl_path, 'rb') as f:
                self.data = pickle.load(f)
        except Exception as e:
            print(f"[ERROR] 加载失败: {e}")
            sys.exit(1)
        
        # 智能解包
        if isinstance(self.data, dict):
            if 'infos' in self.data:
                content = self.data['infos']
                if isinstance(content, list):
                    self.all_frames = content
                elif isinstance(content, dict):
                    for key in content:
                        self.all_frames.extend(content[key])
            else:
                for key in self.data:
                    if isinstance(self.data[key], list):
                        self.all_frames.extend(self.data[key])
        elif isinstance(self.data, list):
            self.all_frames = self.data
        
        print(f"[INFO] 总帧数: {len(self.all_frames)}")
    
    def print_root_summary(self) -> None:
        """以表格形式输出顶层 Key / 类型 / 形状 / 示例内容。"""
        print("\n" + "=" * 120)
        print("顶层结构概览 (Root-Level Summary)")
        print("=" * 120)
        
        if not isinstance(self.data, dict):
            print(f"[INFO] 根对象类型: {type(self.data).__name__}")
            print(self._short_repr(self.data))
            print("=" * 120)
            return
        
        headers = ("Key", "Type", "Shape", "Sample")
        fmt = "{:<30} {:<20} {:<20} {:<100}"
        print(fmt.format(*headers))
        print("-" * 120)
        
        for key in sorted(self.data.keys()):
            val = self.data[key]
            tname = type(val).__name__
            shape_str = self._get_shape_repr(val)
            sample = ""
            # 对于场景列表 / 帧列表，展示长度信息
            if isinstance(val, list):
                sample = f"len={len(val)}; first={self._short_repr(val[0]) if val else 'EMPTY'}"
            else:
                sample = self._short_repr(val)
            print(fmt.format(str(key)[:30], tname[:20], shape_str[:20], sample[:100]))
        
        print("=" * 120)
    
    def print_tree(
        self,
        max_depth: int = 8,
        list_limit: int = 3,
        max_value_len: int = 2000,
        sort_keys: bool = True,
        root_obj: Any = None,
        root_path: str = "root",
    ) -> None:
        """
        以“层级架构表格”方式输出原始 PKL 结构。

        目标：
        - 不做统计/出现率
        - dict 的 Key 不遗漏（会全部列出）
        - list 默认只展开前 list_limit 个元素，但会标注总长度；可用 --list-limit -1 展开全部

        Args:
            max_depth: 最大递归深度
            list_limit: list 展开数量，-1 表示展开全部
            max_value_len: 内容字段最大字符数（超过会截断）
            sort_keys: 是否对 dict keys 排序（稳定输出）
        """

        def iter_dict_keys(d: Dict[str, Any]) -> Iterable[str]:
            keys = list(d.keys())
            if sort_keys:
                try:
                    return sorted(keys)
                except Exception:
                    return keys
            return keys

        print("\n" + "=" * 160)
        print("层级结构树 (Tree View, No-Stats)")
        print("=" * 160)
        fmt = "{:<90} {:<14} {:<24} {:<30}"
        print(fmt.format("Path", "Type", "Shape", "Value (truncated)"))
        print("-" * 160)

        def walk(obj: Any, path: str, depth: int) -> None:
            if depth > max_depth:
                print(fmt.format(path[:90], "...", "...", f"depth>{max_depth}, truncated"))
                return

            tname = type(obj).__name__
            shape = self._get_shape_repr(obj)
            val = self._value_repr(obj, max_len=max_value_len)
            print(fmt.format(path[:90], tname[:14], shape[:24], val[:100]))

            # dict: 列出全部 key，不遗漏
            if isinstance(obj, dict):
                for k in iter_dict_keys(obj):
                    subpath = f"{path}.{k}" if path else str(k)
                    walk(obj[k], subpath, depth + 1)
                return

            # list/tuple: 默认只展开前 N 个，避免刷屏；但不会遗漏“Key”（list 本身无 key）
            if isinstance(obj, (list, tuple)):
                n = len(obj)
                if n == 0:
                    return
                if list_limit == -1:
                    limit = n
                else:
                    limit = min(n, max(0, list_limit))

                for i in range(limit):
                    subpath = f"{path}[{i}]"
                    walk(obj[i], subpath, depth + 1)

                if limit < n:
                    more_path = f"{path}[{limit}..{n-1}]"
                    print(
                        fmt.format(
                            more_path[:90],
                            "...",
                            f"len={n}",
                            f"list truncated, set --list-limit -1",
                        )
                    )
                return

        walk(self.data if root_obj is None else root_obj, root_path, 0)
        print("=" * 160)

    def analyze_structure(self) -> Tuple[str, Any, str]:
        """
        自动检测PKL结构类型
        
        Returns:
            (结构类型, 样本数据, 层级描述)
        """
        struct_type = "Unknown"
        sample_data = None
        desc = ""
        
        # 类型1: 转换后的数据 (List[List[Tuple]]) -> "以场景为中心"
        if isinstance(self.data, list) and len(self.data) > 0:
            if isinstance(self.data[0], list) and len(self.data[0]) > 0:
                first_item = self.data[0][0]
                if isinstance(first_item, tuple) and len(first_item) == 2:
                    struct_type = "Converted (Scene-Centric Sequence)"
                    sample_data = {"Scene_0": self.data[0]}
                    desc = "Root -> List[Scenes] -> List[Frames]"
                    return struct_type, sample_data, desc
        
        # 类型2: 标准 NuScenes 数据 (Dict['infos']) -> "扁平列表"
        if isinstance(self.data, dict) and 'infos' in self.data:
            if isinstance(self.data['infos'], list):
                struct_type = "Standard (Flat Frame List)"
                if len(self.data['infos']) > 0:
                    sample_data = {"infos_sample": self.data['infos'][:1]}
                desc = "Root -> Dict['infos'] -> List[Frames]"
                return struct_type, sample_data, desc
            elif isinstance(self.data['infos'], dict):
                struct_type = "Scene-Centric (Dict[Scene->Frames])"
                sample_data = self.data
                desc = "Root -> Dict['infos'] -> Dict[SceneToken -> List[Frames]]"
                return struct_type, sample_data, desc
        
        # 类型3: 通用字典
        if isinstance(self.data, dict):
            struct_type = "Generic Dictionary"
            sample_data = self.data
            desc = "Root -> Dict"
            return struct_type, sample_data, desc
        
        return struct_type, self.data, "Generic"
    
    def deep_inspect(self, top_n_samples: int = 1) -> None:
        """
        深度表格化检查：统计每个字段的出现率、类型、shape与示例
        
        Args:
            top_n_samples: 每个字段保存多少个示例值
        """
        if len(self.all_frames) == 0:
            print("[WARNING] 没有找到帧数据")
            return
        
        total = len(self.all_frames)
        stats = defaultdict(lambda: {'count': 0, 'types': set(), 'shapes': set(), 'samples': []})
        
        for frame in self.all_frames:
            for k, v in frame.items():
                stat = stats[k]
                stat['count'] += 1
                tname = type(v).__name__
                stat['types'].add(tname)
                
                # 计算shape
                shape_str = 'N/A'
                try:
                    if hasattr(v, 'shape'):
                        shape_str = str(getattr(v, 'shape'))
                    elif isinstance(v, (list, tuple)):
                        if len(v) == 0:
                            shape_str = '(0,)'
                        else:
                            first = v[0]
                            if hasattr(first, 'shape'):
                                shape_str = f'List[{len(v)}]x{first.shape}'
                            else:
                                try:
                                    shape_str = str(np.array(v).shape)
                                except Exception:
                                    shape_str = f'List(len={len(v)})'
                    else:
                        shape_str = 'scalar'
                except Exception:
                    shape_str = 'N/A'
                
                stat['shapes'].add(shape_str)
                
                # 保存示例值
                if len(stat['samples']) < top_n_samples:
                    stat['samples'].append(v)
        
        # 输出表格
        keys_sorted = sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True)
        
        print('\n' + '='*120)
        print('深度检查结果: 字段 | 出现率 | 类型 | 形状 | 示例')
        print('='*120)
        fmt = '{:<30} {:>10} {:<25} {:<25} {:<40}'
        print(fmt.format('字段', '出现率', '类型', '形状', '示例（截断）'))
        print('-'*120)
        
        for k, s in keys_sorted:
            occ = s['count'] / total * 100 if total > 0 else 0
            types_str = '/'.join(sorted(list(s['types']))[:3])
            shapes_str = ' | '.join(list(s['shapes'])[:2])
            sample_repr = ''
            if len(s['samples']) > 0:
                try:
                    sample_repr = repr(s['samples'][0])
                except Exception:
                    sample_repr = str(type(s['samples'][0]))
            if len(sample_repr) > 200:
                sample_repr = sample_repr[:197] + '...'
            
            print(fmt.format(k[:30], f"{occ:6.2f}%", types_str[:25], shapes_str[:25], sample_repr[:40]))
        
        print('='*120 + '\n')
    
    def recursive_traverse(self, obj: Any, path: str, stats: Dict, visited: Set[int], 
                          list_sample_limit: int = 50, depth: int = 0, max_depth: int = 10) -> None:
        """
        递归遍历对象结构
        
        Args:
            obj: 要遍历的对象
            path: 当前路径
            stats: 统计字典
            visited: 已访问对象ID集合（防止循环引用）
            list_sample_limit: 列表采样限制
            depth: 当前深度
            max_depth: 最大深度
        """
        if depth > max_depth:
            return
        
        # 防止循环引用
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)
        
        tname = type(obj).__name__
        rec = stats.setdefault(path, {'count': 0, 'types': set(), 'shapes': set(), 'samples': []})
        rec['count'] += 1
        rec['types'].add(tname)
        
        # 记录shape/sample for leaf-like objects
        if not isinstance(obj, (dict, list)):
            shape_str = self._get_shape_repr(obj)
            rec['shapes'].add(shape_str)
            if len(rec['samples']) < 1:
                try:
                    s = repr(obj)
                except Exception:
                    s = str(type(obj))
                rec['samples'].append(s[:200])
            return
        
        # 对dict: 遍历键
        if isinstance(obj, dict):
            if len(obj) == 0:
                return
            for k, v in obj.items():
                subpath = f"{path}.{k}" if path else str(k)
                self.recursive_traverse(v, subpath, stats, visited, list_sample_limit, depth+1, max_depth)
            return
        
        # 对list: 遍历有限个元素
        if isinstance(obj, list):
            if len(obj) == 0:
                rec['shapes'].add('(0,)')
                return
            rec['shapes'].add(f'List(len={len(obj)})')
            limit = min(len(obj), list_sample_limit)
            for idx in range(limit):
                subpath = f"{path}.[i]" if path else f'[{idx}]'
                self.recursive_traverse(obj[idx], subpath, stats, visited, list_sample_limit, depth+1, max_depth)
            return
    
    def _get_shape_repr(self, v: Any) -> str:
        """获取对象的shape/结构信息。"""
        try:
            # numpy / tensor-like
            if hasattr(v, 'shape'):
                return str(getattr(v, 'shape'))
            # list / tuple
            if isinstance(v, (list, tuple)):
                if len(v) == 0:
                    return '(0,)'
                first = v[0]
                if hasattr(first, 'shape'):
                    return f'List[{len(v)}]x{first.shape}'
                try:
                    return str(np.array(v).shape)
                except Exception:
                    return f'List(len={len(v)})'
            # dict 作为整体：给出 key 数量
            if isinstance(v, dict):
                return f"dict<{len(v)} keys>"
            # 其余当作标量：直接返回类型名（int/float/str/bool 等）
            return type(v).__name__
        except Exception:
            return 'N/A'
    
    def print_field_stats(self, field_name: str) -> None:
        """
        统计字段分布
        
        Args:
            field_name: 字段名
        """
        print("\n" + "="*60)
        print(f"字段统计: {field_name}")
        print("="*60)
        
        counter = Counter()
        missing_count = 0
        
        for frame in self.all_frames:
            if field_name not in frame:
                missing_count += 1
                continue
            
            val = frame[field_name]
            
            if isinstance(val, np.ndarray):
                val_key = tuple(val.tolist())
            elif isinstance(val, list):
                val_key = str(val)
            else:
                val_key = val
            
            try:
                counter[val_key] += 1
            except TypeError:
                counter[str(val_key)] += 1
        
        if missing_count == len(self.all_frames):
            print(f"[WARNING] 字段 '{field_name}' 在所有帧中都不存在")
            return
        
        unique_count = len(counter)
        total_valid = sum(counter.values())
        
        items_to_show = []
        if unique_count > 20:
            print(f"[提醒] 数据种类很多 (唯一值: {unique_count})")
            print(f"[INFO] 仅输出前20个占比最高的数据:")
            items_to_show = counter.most_common(20)
        else:
            items_to_show = counter.most_common()
        
        print("-" * 60)
        for val, count in items_to_show:
            percent = (count / total_valid) * 100 if total_valid > 0 else 0
            val_str = str(val)
            if len(val_str) > 50:
                val_str = val_str[:47] + "..."
            print(f"值: {val_str:<40} | 数量: {count:<6} | {percent:.2f}%")
        
        if unique_count > 20:
            print(f"... (还有 {unique_count - 20} 种数值未显示)")
        
        if missing_count > 0:
            print(f"\n[INFO] 缺失帧数: {missing_count}")
    
    def print_frame_detail(self, frame: Dict, idx: int, target_keys: Optional[List[str]] = None, 
                          title_prefix: str = "Frame") -> None:
        """
        输出单帧详细信息
        
        Args:
            frame: 帧数据
            idx: 帧索引
            target_keys: 要显示的字段列表（None表示显示所有）
            title_prefix: 标题前缀
        """
        print(f"\n--- {title_prefix} {idx} 详情 ---")
        
        keys_to_iter = target_keys if (target_keys and len(target_keys) > 0) else frame.keys()
        
        for key in keys_to_iter:
            if key not in frame:
                print(f"键: {key:<25} | [WARNING] 字段不存在")
                continue
            
            val = frame[key]
            val_type = type(val).__name__
            val_shape = "N/A"
            
            if isinstance(val, (np.ndarray, list, tuple)):
                try:
                    val_shape = str(np.array(val).shape)
                except:
                    val_shape = f"len={len(val)}"
            
            val_str = str(val)
            if len(val_str) > 2000:
                val_str = val_str[:2000] + " ... [截断]"
            
            print(f"键: {key:<25} | 类型: {val_type:<10} | 形状: {val_shape:<12} | 内容: {val_str}")
    
    def analyze_weather_distribution(self, keywords_dict: Dict[str, List[str]], 
                                    desc_field: str = 'description') -> Tuple[Counter, List[Dict]]:
        """
        统计天气关键词分布
        
        Args:
            keywords_dict: 天气关键词字典 {天气类型: [关键词列表]}
            desc_field: 包含天气描述的字段名
        
        Returns:
            (天气计数器, 匹配的帧列表)
        """
        print("\n" + "="*80)
        print(f"天气分布分析 (基于字段 '{desc_field}')")
        print("="*80)
        
        weather_counter = Counter()
        matched_frames = []
        unknown_descriptions = []
        
        for frame_idx, frame in enumerate(self.all_frames):
            if desc_field not in frame:
                continue
            
            desc = str(frame[desc_field]).lower()
            matched_weather = None
            
            # 按优先级匹配天气类型
            for weather_type, keywords in keywords_dict.items():
                if any(keyword in desc for keyword in keywords):
                    matched_weather = weather_type
                    break
            
            if matched_weather is None:
                matched_weather = 'unknown'
                unknown_descriptions.append({
                    'frame_idx': frame_idx,
                    'description': frame[desc_field]
                })
            
            weather_counter[matched_weather] += 1
            matched_frames.append({
                'frame_idx': frame_idx,
                'weather': matched_weather,
                'description': frame[desc_field]
            })
        
        # 输出统计结果
        total = len(matched_frames)
        print(f"\n总分析帧数: {total}")
        print("\n" + "-"*80)
        print("天气分布:")
        print("-"*80)
        
        for weather_type, count in weather_counter.most_common():
            percentage = (count / total * 100) if total > 0 else 0
            bar_length = int(percentage / 2)
            bar = '█' * bar_length
            print(f"{weather_type:<20} | 数量: {count:<6} | {percentage:>6.2f}% | {bar}")
        
        # 如果有unknown的数据，输出示例
        if unknown_descriptions:
            print("\n" + "-"*80)
            print(f"未知描述 (总数: {len(unknown_descriptions)}, 显示前10个):")
            print("-"*80)
            for item in unknown_descriptions[:10]:
                print(f"  帧 {item['frame_idx']}: {item['description']}")
        
        print("\n" + "="*80 + "\n")
        return weather_counter, matched_frames
    
    def compare_fields(self, field_a: str, field_b: str) -> None:
        """
        对比两个字段的值
        
        Args:
            field_a: 字段A
            field_b: 字段B
        """
        print("\n" + "#"*80)
        print(f">>> 字段对比: {field_a} vs {field_b}")
        print("#"*80)
        
        diff_count = 0
        for i, frame in enumerate(self.all_frames):
            if field_a not in frame or field_b not in frame:
                continue
            val_a, val_b = frame[field_a], frame[field_b]
            is_same = False
            if isinstance(val_a, np.ndarray) and isinstance(val_b, np.ndarray):
                if val_a.shape == val_b.shape:
                    is_same = np.array_equal(val_a, val_b)
            else:
                is_same = (val_a == val_b)
            if not is_same:
                diff_count += 1
                if diff_count <= 20:
                    print(f"帧 {i}: {field_a}={val_a} != {field_b}={val_b}")
        
        if diff_count == 0:
            print("[SUCCESS] 所有值都匹配！")
        else:
            print(f"[WARNING] 总不匹配帧数: {diff_count}")
    
    def convert_key(self, old_key: str, new_key: str, output_path: Optional[str] = None) -> None:
        """
        转换PKL文件中的键名
        
        Args:
            old_key: 旧键名（支持嵌套，如 'cams.cam_intrinsic'）
            new_key: 新键名（支持嵌套，如 'cams.camera_intrinsics'）
            output_path: 输出路径（None则覆盖原文件）
        """
        print(f"[INFO] 转换键名: {old_key} -> {new_key}")
        
        def convert_in_dict(d: Dict, old: str, new: str) -> int:
            """递归转换字典中的键"""
            count = 0
            if '.' in old:
                # 嵌套键
                parts = old.split('.', 1)
                if parts[0] in d and isinstance(d[parts[0]], dict):
                    count += convert_in_dict(d[parts[0]], parts[1], new.split('.', 1)[1] if '.' in new else new)
            else:
                # 直接键
                if old in d:
                    d[new] = d.pop(old)
                    count = 1
            return count
        
        converted_count = 0
        if isinstance(self.data, dict):
            if 'infos' in self.data:
                if isinstance(self.data['infos'], list):
                    for frame in self.data['infos']:
                        converted_count += convert_in_dict(frame, old_key, new_key)
                elif isinstance(self.data['infos'], dict):
                    for scene_frames in self.data['infos'].values():
                        for frame in scene_frames:
                            converted_count += convert_in_dict(frame, old_key, new_key)
        
        print(f"[INFO] 转换了 {converted_count} 个键")
        
        output_path = output_path or self.pkl_path
        print(f"[INFO] 保存到: {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(self.data, f)
        print("[INFO] 完成！")


def main():
    parser = argparse.ArgumentParser(
        description="PKL文件检查和分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 层级树（不统计，不打散结构；dict 的 key 不遗漏）
  python inspect_pkl.py data/nuscenes_mini_infos_temporal_train_scene.pkl --mode tree --max-depth 6 --list-limit 2
  python inspect_pkl.py data/nuscenes_mini_infos_temporal_train_scene.pkl --mode tree --max-depth 12 --list-limit 5 --max-value-len 500 
  python inspect_pkl.py data/nuscenes_mini_infos_temporal_train_scene.pkl --mode deep --max-depth 5
  
  # 只展开某个 scene 的第 N 帧（避免扫全量）——适用于 infos 是 dict[scene]->list[frames] 的 temporal PKL
  python inspect_pkl.py data/nuscenes_mini_infos_temporal_train_scene.pkl --mode tree --scene scene-0061 --frame-idx 0 --max-depth 12 --list-limit -1 --max-value-len 2000
  
  # 12Hz frame-list PKL（infos 是 list），用全局帧索引
  python inspect_pkl.py data/nuscenes_mmdet3d-12Hz/nuscenes_mini_advanced_12Hz_infos_train.pkl --mode tree --global-frame-idx 0 --max-depth 12 --list-limit -1 --max-value-len 2000
  
  # 字段统计
  python inspect_pkl.py data/infos.pkl --mode stats --keys token timestamp
  
  # 天气分布分析
  python inspect_pkl.py data/infos.pkl --mode weather
  
  # 字段对比
  python inspect_pkl.py data/infos.pkl --mode compare --field-a gt_ego_fut_cmd --field-b pose_mode
  
  # 键名转换
  python inspect_pkl.py data/infos.pkl --mode convert --old-key cam_intrinsic --new-key camera_intrinsics
        """
    )
    
    parser.add_argument('pkl_path', type=str, help='PKL文件路径')
    parser.add_argument('--mode', type=str, default='deep',
                       choices=['tree', 'deep', 'stats', 'weather', 'compare', 'convert', 'summary'],
                       help='检查模式: deep=深度结构分析, stats=字段统计, weather=天气分析, '
                            'compare=字段对比, convert=键名转换, summary=快速摘要')
    parser.add_argument('--max-depth', type=int, default=10, help='递归遍历最大深度（deep模式）')
    parser.add_argument(
        '--list-limit',
        type=int,
        default=3,
        help='tree 模式下 list/tuple 展开前 N 个元素；-1 表示展开全部（可能很长）',
    )
    parser.add_argument(
        '--max-value-len',
        type=int,
        default=2000,
        help='tree 模式下 Value 字段最大字符数（超过会截断）',
    )
    parser.add_argument(
        '--scene',
        type=str,
        default=None,
        help="tree 模式下：只展开 infos[scene] 下的某一帧（scene->frames 结构专用，例如 scene-0061）",
    )
    parser.add_argument(
        '--frame-idx',
        type=int,
        default=None,
        help="tree 模式下：与 --scene 配合使用，指定该 scene 内的第几帧（从 0 开始）",
    )
    parser.add_argument(
        '--global-frame-idx',
        type=int,
        default=None,
        help="tree 模式下：基于 all_frames 的全局帧索引（适用于 infos 是 list 的 12Hz PKL）",
    )
    parser.add_argument('--keys', type=str, nargs='+', help='要统计的字段列表（stats模式）')
    parser.add_argument('--field-a', type=str, help='对比字段A（compare模式）')
    parser.add_argument('--field-b', type=str, help='对比字段B（compare模式）')
    parser.add_argument('--old-key', type=str, help='旧键名（convert模式）')
    parser.add_argument('--new-key', type=str, help='新键名（convert模式）')
    parser.add_argument('--output', type=str, help='输出文件路径（convert模式）')
    parser.add_argument('--dump-rows', type=int, default=0, help='输出前N帧详情（summary模式）')
    parser.add_argument('--dump-keys', type=str, nargs='+', help='要输出的字段列表（summary模式）')
    
    args = parser.parse_args()
    
    inspector = PKLInspector(args.pkl_path)
    
    # 所有模式前先输出顶层结构表格，方便快速了解根级变量名/类型/形状/内容
    inspector.print_root_summary()
    
    if args.mode == 'tree':
        root_obj = None
        root_path = "root"

        # 优先：基于 scene + frame-idx 的定点展开（仅适用于 infos 是 dict[scene]->list[frames]）
        if args.scene is not None or args.frame_idx is not None:
            if args.scene is None or args.frame_idx is None:
                print("[ERROR] tree 模式下使用定点展开需要同时提供 --scene 和 --frame-idx")
                sys.exit(2)
            if not isinstance(inspector.data, dict) or 'infos' not in inspector.data:
                print("[ERROR] 当前 PKL 不是 dict['infos'] 结构，无法使用 --scene/--frame-idx")
                sys.exit(2)
            infos = inspector.data['infos']
            if not isinstance(infos, dict):
                print("[ERROR] 当前 PKL 的 infos 不是 dict（不是 scene->frames），无法使用 --scene/--frame-idx")
                sys.exit(2)
            if args.scene not in infos:
                available = list(infos.keys())
                preview = ", ".join(sorted(available)[:10])
                print(f"[ERROR] scene 不存在: {args.scene}")
                print(f"[INFO] 可用 scene 示例（前 10 个）: {preview}")
                sys.exit(2)
            frames = infos[args.scene]
            if not isinstance(frames, list):
                print(f"[ERROR] infos[{args.scene}] 不是 list，实际类型: {type(frames).__name__}")
                sys.exit(2)
            if args.frame_idx < 0 or args.frame_idx >= len(frames):
                print(f"[ERROR] frame-idx 越界: {args.frame_idx}，该 scene 帧数={len(frames)}")
                sys.exit(2)
            root_obj = frames[args.frame_idx]
            root_path = f"root.infos.{args.scene}[{args.frame_idx}]"

        # 次选：基于 all_frames 的全局帧索引（适用于 infos 是 list 的 12Hz PKL）
        elif args.global_frame_idx is not None:
            if len(inspector.all_frames) == 0:
                print("[ERROR] all_frames 为空，无法使用 --global-frame-idx")
                sys.exit(2)
            if args.global_frame_idx < 0 or args.global_frame_idx >= len(inspector.all_frames):
                print(
                    f"[ERROR] global-frame-idx 越界: {args.global_frame_idx}，"
                    f"all_frames 总帧数={len(inspector.all_frames)}"
                )
                sys.exit(2)
            root_obj = inspector.all_frames[args.global_frame_idx]
            root_path = f"root.all_frames[{args.global_frame_idx}]"

        inspector.print_tree(
            max_depth=args.max_depth,
            list_limit=args.list_limit,
            max_value_len=args.max_value_len,
            root_obj=root_obj,
            root_path=root_path,
        )
        print("\n[完成]")
        return

    if args.mode == 'deep':
        # 深度结构分析
        struct_type, sample_data, hierarchy_desc = inspector.analyze_structure()
        print("-" * 140)
        print(f"结构类型: {struct_type}")
        print(f"层级描述: {hierarchy_desc}")
        print("-" * 140)
        
        # 递归遍历
        stats = {}
        visited = set()
        inspector.recursive_traverse(inspector.data, '', stats, visited, max_depth=args.max_depth)
        
        # 计算出现率
        total_encounters = sum(v['count'] for v in stats.values()) if len(stats) > 0 else 1
        frames_total = len(inspector.all_frames)
        
        print('\n' + '='*140)
        print('PKL字段深度探测结果（路径 | 出现率 | 类型 | 形状 | 示例）')
        print('='*140)
        fmt = '{:<60} {:>7} {:<25} {:<20} {:<40}'
        print(fmt.format('路径', '出现率', '类型', '形状', '示例'))
        print('-'*140)
        
        for path, info in sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True):
            count = info['count']
            if frames_total > 0 and '.[i]' in path:
                occ = count / frames_total * 100
            else:
                occ = count / total_encounters * 100
            
            types_str = '/'.join(sorted(list(info['types']))[:3])
            shapes_str = ' | '.join(list(info['shapes'])[:2])
            example = info['samples'][0] if len(info['samples']) > 0 else ''
            if len(example) > 200:
                example = example[:197] + '...'
            
            print(fmt.format(path[:60], f"{occ:6.2f}%", types_str[:25], shapes_str[:20], example[:40]))
        
        print('='*140)
    
    elif args.mode == 'stats':
        # 字段统计
        inspector.deep_inspect()
        if args.keys:
            for key in args.keys:
                inspector.print_field_stats(key)
    
    elif args.mode == 'weather':
        # 天气分布分析
        weather_keywords = {
            'rain': ['rain', 'raining', 'rainy', 'wet'],
            'heavy rain': ['heavy rain', 'pouring'],
            'night': ['night', 'nighttime', 'dark'],
            'day': ['day', 'daytime', 'sunny', 'clear'],
            'overcast': ['overcast', 'cloudy'],
            'fog': ['fog', 'foggy'],
            'snow': ['snow', 'snowing'],
        }
        inspector.analyze_weather_distribution(weather_keywords)
    
    elif args.mode == 'compare':
        # 字段对比
        if not args.field_a or not args.field_b:
            print("[ERROR] compare模式需要指定 --field-a 和 --field-b")
            sys.exit(1)
        inspector.compare_fields(args.field_a, args.field_b)
    
    elif args.mode == 'convert':
        # 键名转换
        if not args.old_key or not args.new_key:
            print("[ERROR] convert模式需要指定 --old-key 和 --new-key")
            sys.exit(1)
        inspector.convert_key(args.old_key, args.new_key, args.output)
    
    elif args.mode == 'summary':
        # 快速摘要
        struct_type, sample_data, hierarchy_desc = inspector.analyze_structure()
        print(f"\n结构类型: {struct_type}")
        print(f"层级描述: {hierarchy_desc}")
        print(f"总帧数: {len(inspector.all_frames)}")
        
        if args.dump_rows > 0:
            dump_keys = args.dump_keys if args.dump_keys else None
            limit = min(len(inspector.all_frames), args.dump_rows)
            for i in range(limit):
                inspector.print_frame_detail(inspector.all_frames[i], i, target_keys=dump_keys)
    
    print("\n[完成]")


if __name__ == "__main__":
    main()

"""
python inspect_pkl.py data/nuscenes_mini_infos_temporal_train_scene_v2.pkl   --mode tree --scene scene-0061 --frame-idx 0   --max-depth 12 --list-limit -1 --max-value-len 2000
"""