#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
兼容入口：转发到 `dataset_preprocess.inspect_pkl` 的主函数。

这样你既可以：
  python dataset_preprocess/inspect_pkl.py ...
也可以：
  python inspect_pkl.py ...
两者行为完全一致，不再依赖硬编码的 pkl 路径。
"""

from dataset_preprocess.inspect_pkl import main


if __name__ == "__main__":
    main()