# UniData (Vectorized Autonomous Driving) 环境配置指南

**摘要**：本文旨在记录如何在 Linux 环境下（Ubuntu 22.04）复现 unidata 算法，并集成 ASAP 项目的数据处理模块（插帧功能）。鉴于 OpenMMLab 系列库（mmdet, mmseg, mmcv）与 CUDA、PyTorch 之间存在严格的版本依赖关系，且本环境需同时满足 VAD 的基础运行与 ASAP 的稀疏卷积需求，请务必严格按照以下步骤顺序执行，切勿擅自升级核心库版本。

**前置条件**：
*   OS: Ubuntu 22.04/20.04 (推荐)
*   GPU Driver: 适配 CUDA 11.1 的驱动版本
*   Anaconda / Miniconda 已安装

---

### 第一步：创建并激活虚拟环境

为了保证底层兼容性，选用 Python 3.8 版本。

```bash
conda create -n unidata python=3.8 -y
conda activate unidata
```

### 第二步：安装 PyTorch 与 CUDA 运行时

需安装 PyTorch 1.9.1 配合 CUDA 11.1。

```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### 第三步：安装 OpenMMLab 基础依赖库

UniData 项目严格锁定了 `mmcv-full`, `mmdet` 和 `mmsegmentation` 的版本。在安装具体库之前，首先安装 `openmim` 管理工具，尽管我们将手动指定版本以确保精确性。

1.  **安装 OpenMIM**
    ```bash
    pip install openmim
    ```

2.  **安装 MMCV-Full**
    注意：必须安装 1.4.0 版本。
    ```bash
    pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.1/index.html
    ```

3.  **安装 MMDetection 与 MMSegmentation**
    严禁升级这两个库，否则会导致 UniData 代码中的 API 调用失败。
    ```bash
    pip install mmdet==2.14.0
    pip install mmsegmentation==0.14.1
    ```

4.  **安装 Timm**
    ```bash
    pip install timm
    ```

### 第四步：编译安装 MMDetection3D

为了便于调试和集成，我们将 `mmdetection3d` 作为源码编译安装在 UniData 项目目录下。

1.  **克隆 UniData 仓库（若未克隆）及准备目录**
    ```bash
    # 假设当前在工作空间根目录
    git clone https://github.com/hustvl/UniData.git
    cd UniData
    ```

2.  **克隆并编译 MMDetection3D v0.17.1**
    注意：请确保 `mmdetection3d` 目录位于 `UniData` 根目录下。
    ```bash
    # 在 UniData 目录下执行
    git clone https://github.com/open-mmlab/mmdetection3d.git
    cd mmdetection3d
    git checkout -f v0.17.1
    
    # 安装构建依赖（预防性安装）
    pip install setuptools==60.2.0 
    
    # 源码编译安装
    pip install -v -e .
    ```

### 第五步：安装 ASAP 数据处理相关依赖

为了在 UniData 环境中运行插帧代码，需要安装稀疏卷积库 `spconv`。针对 CUDA 11.1 环境，需选择以下特定版本：

```bash
pip install spconv-cu111==2.1.25 cumm-cu111==0.2.9
```

### 第六步：解决关键依赖冲突（重点）

在实际配置过程中，会遇到 `numpy` 版本导致的 `np.long` 报错以及 `requests` 库的版本冲突。请按以下方案修正。

#### 1. 修正 Numpy 版本（解决 AttributeError: module 'numpy' has no attribute 'long'）
新版 Numpy（1.24+）移除了 `np.long` 类型，导致旧版 MMLab 代码报错。同时，直接使用 pip 降级可能会破坏 conda 的底层数学库连接（如 mkl/blas）。

**解决方案**：使用 Conda 强制重装指定版本的 Numpy 及图像库。

```bash
# 这一步会同时修复 numpy 版本和找回可能丢失的底层库
conda install numpy=1.19.5 pillow imageio -y
```

#### 2. 修正 Requests 版本（解决 OpenXLab 依赖冲突）
`openxlab` 需要旧版 `requests`，而 Jupyter 相关组件通常需要新版。在本项目中，优先保障代码运行环境。

**解决方案**：降级 requests。

```bash
pip install requests==2.28.2
```
*注：执行此命令后，若出现关于 `jupyterlab-server` 依赖冲突的红色报错，请直接忽略。只要终端显示 `Successfully installed requests-2.28.2` 即视为成功。该冲突不影响 UniData 和 ASAP 核心代码的运行。*

### 第七步：安装其他必要工具

安装 NuScenes 数据集开发工具包。

```bash
pip install nuscenes-devkit==1.1.9
```

### 第八步：验证环境

完成上述步骤后，可使用 `pip list` 检查关键包版本是否符合预期：

*   `torch`: 1.9.1+cu111
*   `mmcv-full`: 1.4.0
*   `mmdet`: 2.14.0
*   `mmsegmentation`: 0.14.1
*   `mmdetection3d`: 0.17.1 (路径应指向 UniData/mmdetection3d)
*   `numpy`: 1.19.5
*   `spconv-cu111`: 2.1.25

至此，UniData 环境配置完成，且已包含运行 ASAP 数据处理模块所需的所有依赖。可以开始进行预训练模型准备及数据处理流程。