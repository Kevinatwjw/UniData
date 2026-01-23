#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 nuScenes 2Hz / 12Hz PKL 离线添加场景级天气与昼夜标签。

功能概述
--------
- 支持两类 PKL 结构：
  1) 2Hz temporal 格式（unidata_nuscenes_converter.py 生成）
     - 扁平列表:   data['infos'] 是 list[frame_dict]
     - 场景格式:   data['infos'] 是 dict[scene_id] -> list[frame_dict]
  2) 12Hz list 格式（unidata_nuscenes_12Hz_converter.py 生成）
     - data = {'infos': list[frame_dict], 'metadata': ..., 'scene_tokens': List[List[token]]}

- 每个「场景」增加两个字段（写回到每一帧的 info 上）：
  - 'weather':  'sunny' 或 'rainy'
  - 'daytime':  'day' 或 'night'

计算规则
--------
1. weather（晴/雨，基于 description 文本）
   - 读取该场景第一帧的 `description`，转小写；
   - 如果命中 rainy 关键词 → weather='rainy'
   - 否则一律视为 weather='sunny'（包括 clear/overcast/fog 等情况）

2. daytime（昼/夜，优先 description，其次天文计算）
   - 先用 description 匹配：
       * 若命中 night 关键词 → 'night'
       * 否则若命中 day 或 sunny 关键词 → 'day'
   - 若 description 无法判定：
       * 使用帧的 `timeofday`（如 '2018-07-24-11-22-45+0800'）和 `location`
       * 通过 astral + pytz 按经纬度与时区计算该时刻是否在日出~日落区间
       * 若无法计算（缺少依赖或字段），默认 'day'

3. 场景分组方式
   - 如果 data['infos'] 是 dict：直接把每个 key 当作一个 scene（如 'scene-0061'）
   - 如果 data['infos'] 是 list：
       * 优先使用 data['scene_tokens'] (List[List[token]]) 进行分组
       * 否则如果每帧包含 'scene_name' 或 'scene_token'，按该字段分组
       * 如果都没有，就将整份 infos 当成一个大场景处理

4. 输出
   - 不改变原有结构，只是在每个 frame 的字典中增加两个 key：
       * frame['weather'] = 'sunny' | 'rainy'
       * frame['daytime'] = 'day'   | 'night'
   - 输出文件路径：
       * 若未指定 --out，则在原文件名后自动加 `_v2.pkl`

使用示例
--------
1) 处理 2Hz temporal scene 格式 PKL（infos 为 dict[scene]->list[frames]）:

   python tools/data_converter/nuscenes_weather_daytime_extractor.py data/nuscenes_mini_infos_temporal_train_scene.pkl
   python tools/data_converter/nuscenes_weather_daytime_extractor.py data/nuscenes_mini_infos_temporal_val_scene.pkl

2) 处理 2Hz 扁平 temporal PKL（infos 为 list[frames]）:

   python tools/data_converter/nuscenes_weather_daytime_extractor.py \\
       data/nuscenes_mini_infos_temporal_train.pkl

3) 处理 12Hz advanced list PKL（infos 为 list[frames] + scene_tokens）:

   python tools/data_converter/nuscenes_weather_daytime_extractor.py \\
       data/nuscenes_mmdet3d-12Hz/nuscenes_mini_advanced_12Hz_infos_train.pkl
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple
import mmcv
import numpy as np
from astral import LocationInfo  # type: ignore
from astral.sun import sun as astral_sun  # type: ignore
import pytz  # type: ignore

_ASTRAL_AVAILABLE = True


logger = logging.getLogger(__name__)


WEATHER_KEYWORDS: Dict[str, List[str]] = {
    "rainy": [
        # 原始 + 补充
        "rain", "raining", "rainy", "wet", "heavy rain", "pouring",
        "drizzle", "light rain", "shower", "showers", "downpour", "rainfall",
        "rain drops", "rainy conditions", "rain weather", "wet road", "puddles",
        "rainy day", "rain at night",
    ],
    "night": [
        "night", "nighttime", "dark", "dusk", "twilight",
        "evening", "late evening", "night scene", "dark conditions", "low light",
        "street lights", "night driving", "after sunset", "dark outside", "night time",
    ],
    "day": [
        "day", "daytime", "clear",
        "bright", "daylight", "sunlight",
        "overcast", "cloudy", "partly cloudy", "fair weather", "good weather",
        "clear sky", "sunny conditions", "day scene", "morning", "afternoon",
    ],
    "sunny": [
        "sunny", "sunny day", "clear day", "sunshine", "bright sunshine",
        "sunny weather", "clear sunny", "bright and sunny", "sunny conditions",
        "good sunshine", "sunny afternoon", "sunny morning", "clear and sunny",
        "sunny scene", "bright sunlight", "full sun", "sunny skies", "no clouds",
        "blue sky",
    ],
}

# 简单的 location -> (lat, lon, timezone) 映射
LOCATION_INFO: Dict[str, Tuple[float, float, str]] = {
    "singapore-onenorth": (1.29, 103.78, "Asia/Singapore"),
    "singapore-queenstown": (1.28, 103.80, "Asia/Singapore"),
    "singapore-hollandvillage": (1.31, 103.79, "Asia/Singapore"),
    "boston-seaport": (42.35, -71.04, "America/New_York"),
}


def _normalize_str(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def classify_weather(description: str) -> str:
    """根据文本描述粗略判断晴/雨."""
    desc = _normalize_str(description)
    if any(kw in desc for kw in WEATHER_KEYWORDS["rainy"]):
        return "rainy"
    # 其余一律认为是晴天（包含多云/阴天）
    return "sunny"


def classify_daytime_from_description(description: str) -> Optional[str]:
    """尝试仅根据 description 判定昼夜；返回 'day'/'night' 或 None."""
    desc = _normalize_str(description)
    if any(kw in desc for kw in WEATHER_KEYWORDS["night"]):
        return "night"
    if any(kw in desc for kw in WEATHER_KEYWORDS["day"]) or any(
        kw in desc for kw in WEATHER_KEYWORDS["sunny"]
    ):
        return "day"
    return None


def classify_daytime_from_astral(
    location: Optional[str],
    time_of_day: Optional[str],
) :  # -> Optional[str]
    """使用 astral + pytz 根据地理位置和时间戳计算昼夜.

    time_of_day 格式通常为 'YYYY-mm-dd-HH-MM-SS+ZZZZ' (例如 '2018-07-24-11-22-45+0800')
    """
    if not _ASTRAL_AVAILABLE:
        logger.debug("astral/pytz 未安装，跳过天文昼夜计算，回退为 None")
        return None

    if not location or not time_of_day:
        return None

    loc = _normalize_str(location)
    if loc not in LOCATION_INFO:
        logger.debug("未知 location=%s，无法进行天文计算", location)
        return None

    lat, lon, tz_name = LOCATION_INFO[loc]

    try:
        # 解析带时区偏移的时间
        dt = _dt.datetime.strptime(time_of_day, "%Y-%m-%d-%H-%M-%S%z")
    except ValueError:
        logger.warning("无法解析 timeofday '%s'，期望格式类似 2018-07-24-11-22-45+0800", time_of_day)
        return None

    try:
        tz = pytz.timezone(tz_name)
        local_dt = dt.astimezone(tz)
        loc_info = LocationInfo(name=loc, region="", timezone=tz_name, latitude=lat, longitude=lon)
        s = astral_sun(loc_info.observer, date=local_dt.date(), tzinfo=tz)
        sunrise = s["sunrise"]
        sunset = s["sunset"]
        if sunrise <= local_dt <= sunset:
            return "day"
        return "night"
    except Exception as e:  # pragma: no cover
        logger.warning("astral 计算昼夜失败: %s，回退为 None", e)
        return None


def classify_daytime(description: str, location: Optional[str], time_of_day: Optional[str]) -> str:
    """综合 description 与天文计算判断昼夜."""
    from_desc = classify_daytime_from_description(description)
    if from_desc is not None:
        return from_desc

    from_sun = classify_daytime_from_astral(location, time_of_day)
    if from_sun is not None:
        return from_sun

    # 兜底：白天
    return "day"


def compute_scene_labels(example: Mapping[str, Any]) -> Tuple[str, str]:
    """从单帧示例中提取 description/location/timeofday，并计算 (weather, daytime)."""
    desc = str(example.get("description", "") or "")
    loc = example.get("location") or example.get("map_location")
    tstr = example.get("timeofday")
    weather = classify_weather(desc)
    daytime = classify_daytime(desc, loc, tstr)
    return weather, daytime


def iter_scenes_from_infos_dict(
    infos: Mapping[str, List[MutableMapping[str, Any]]]
) -> Iterable[Tuple[str, List[MutableMapping[str, Any]]]]:
    """场景字典形式: infos: Dict[scene_id, List[frames]]."""
    for scene_id, frames in infos.items():
        if not frames:
            continue
        yield scene_id, frames


def iter_scenes_from_flat_list(
    data: Mapping[str, Any],
) -> Iterable[Tuple[str, List[MutableMapping[str, Any]]]]:
    """从扁平 infos:list 结构恢复场景分组."""
    infos = data["infos"]

    # 1) 优先使用 scene_tokens (List[List[token]])
    scene_tokens = data.get("scene_tokens")
    if isinstance(scene_tokens, list) and infos and isinstance(infos[0], Mapping):
        token_to_idx: Dict[str, int] = {
            str(frm.get("token")): idx for idx, frm in enumerate(infos) if "token" in frm
        }
        for idx, token_list in enumerate(scene_tokens):
            if not isinstance(token_list, list) or not token_list:
                continue
            frames: List[MutableMapping[str, Any]] = []
            for tok in token_list:
                key = str(tok)
                if key in token_to_idx:
                    frames.append(infos[token_to_idx[key]])
            if not frames:
                continue
            scene_id = f"scene_{idx}"
            yield scene_id, frames
        return

    # 2) 其次：若每帧有 scene_name 或 scene_token，则按该字段分组
    key_name = None
    if infos and all(isinstance(f, Mapping) and "scene_name" in f for f in infos):
        key_name = "scene_name"
    elif infos and all(isinstance(f, Mapping) and "scene_token" in f for f in infos):
        key_name = "scene_token"

    if key_name is not None:
        groups: Dict[str, List[MutableMapping[str, Any]]] = {}
        for frm in infos:
            sid = str(frm.get(key_name))
            groups.setdefault(sid, []).append(frm)
        for sid, frames in groups.items():
            if not frames:
                continue
            yield sid, frames
        return

    # 3) 最后：无法区分场景时，把整个列表当作一个场景
    if infos:
        yield "all", infos


def annotate_pkl(data: MutableMapping[str, Any]) -> None:
    """根据 PKL 的结构，为其中的帧添加 weather/daytime 字段."""
    infos = data.get("infos")
    if infos is None:
        raise ValueError("输入 PKL 缺少顶层键 'infos'")

    # 场景级迭代器
    if isinstance(infos, dict):
        scene_iter = iter_scenes_from_infos_dict(infos)
    elif isinstance(infos, list):
        scene_iter = iter_scenes_from_flat_list(data)
    else:
        raise ValueError(f"暂不支持的 infos 类型: {type(infos)} (期望 list 或 dict)")

    scene_count = 0
    frame_count = 0

    for scene_id, frames in scene_iter:
        scene_count += 1
        example = frames[0]
        weather, daytime = compute_scene_labels(example)
        logger.info(
            "Scene %s: weather=%s, daytime=%s (desc='%s')",
            scene_id,
            weather,
            daytime,
            str(example.get("description", ""))[:80],
        )
        for frm in frames:
            # 不覆盖已有字段（如果已经有人手动写过）
            frm.setdefault("weather", weather)
            frm.setdefault("daytime", daytime)
            frame_count += 1

    logger.info("共处理场景数: %d, 帧数: %d", scene_count, frame_count)


def build_out_path(in_path: str, out_path: Optional[str]) -> str:
    """根据输入路径生成输出路径（默认加 `_v2.pkl` 后缀）."""
    if out_path:
        return out_path
    base, ext = os.path.splitext(in_path)
    if not ext:
        ext = ".pkl"
    return f"{base}_v2{ext}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为 nuScenes 2Hz / 12Hz PKL 添加 weather/daytime 字段",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=str,
        help="输入 PKL 文件路径（2Hz 或 12Hz 的 infos*.pkl / *_scene.pkl）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出 PKL 路径（默认在文件名后自动加 _v2）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印更详细的日志信息",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    in_path = args.input
    out_path = build_out_path(in_path, args.out)

    logger.info("加载 PKL: %s", in_path)
    data = mmcv.load(in_path)

    annotate_pkl(data)

    logger.info("保存带 weather/daytime 的新 PKL 至: %s", out_path)
    mmcv.dump(data, out_path)
    logger.info("完成.")


if __name__ == "__main__":
    main()

