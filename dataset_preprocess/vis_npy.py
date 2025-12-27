import pickle
import numpy as np
import matplotlib.pyplot as plt

# 读取你的 .npy (pickle)
file_path = "data/GT_occupancy_mini_12Hz/dense_voxels_with_semantic/0a0d6b8c2e884134a3b48df43d54c36a1/3e92d615685942dfab6edb5a0a7678e61.npy"
with open(file_path, "rb") as f:
    data = pickle.load(f) # (N, 4)

# data[:, :3] 是坐标
# 简单的切片可视化 (查看 Z=3 的平面)
slice_z = data[data[:, 2] == 3]
plt.scatter(slice_z[:, 0], slice_z[:, 1], s=1)
plt.show()