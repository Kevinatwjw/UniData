# UniData 环境搭建指南

**简介**：本指南记录了如何在 Linux 环境（推荐 Ubuntu 22.04）下复现 UniData 算法，并集成了 ASAP 项目的数据处理模块（用于帧插值）。鉴于 OpenMMLab 系列库（mmdet, mmseg, mmcv）、CUDA 和 PyTorch 之间存在严格的版本依赖关系，且该环境需同时支持 VAD 的基础运行以及 ASAP 的稀疏卷积需求，请务必严格按照以下步骤顺序执行。**切勿随意升级核心库的版本。**

**前置条件**：

*   **操作系统**：Ubuntu 22.04 / 20.04 (推荐)
*   **显卡驱动**：版本需兼容 CUDA 11.1
*   **Anaconda / Miniconda**：已安装

-----

### 第一步：创建并激活虚拟环境

为确保底层库的兼容性，我们使用 Python 3.8。

```bash
conda create -n unidata python=3.8 -y
conda activate unidata
```

### 第二步：安装 PyTorch 和 CUDA 运行时

安装适配 CUDA 11.1 的 PyTorch 1.9.1。

```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### 第三步：安装 OpenMMLab 基础依赖

UniData 项目严格锁定了 `mmcv-full`、`mmdet` 和 `mmsegmentation` 的版本。在安装具体库之前，先安装 `openmim` 管理工具（尽管我们会手动指定版本以确保精确性）。

1.  **安装 OpenMIM**

    ```bash
    pip install openmim
    ```

2.  **安装 MMCV-Full**
    **注意**：必须是 1.4.0 版本。

    ```bash
    pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.1/index.html
    ```

    **故障排除：**
    如果安装过程卡在 **"Building wheel for mmcv-full"**，说明 `pip` 未能找到兼容的预编译包，正在尝试从源码编译（这通常极慢且容易报错）。

    **解决方案：**
    按 `Ctrl + C` 终止当前进程。不要直接使用 pip，改用 `openmim` 自动检测并安装正确的预编译版本：

    ```bash
    mim install mmcv-full==1.4.0
    ```

3.  **安装 MMDetection 和 MMSegmentation**
    **严禁升级**这两个库，否则 UniData 代码中的 API 调用将会失效。

    ```bash
    pip install mmdet==2.14.0
    pip install mmsegmentation==0.14.1
    ```

4.  **安装 Timm**

    ```bash
    pip install timm
    ```

### 第四步：编译安装 MMDetection3D

为了便于调试和代码集成，我们将直接在 UniData 项目目录下从源码编译安装 `mmdetection3d`。

1.  **克隆 UniData 仓库（如未克隆）并进入目录**

    ```bash
    # 假设您在工作区根目录
    git clone https://github.com/Kevinatwjw/UniData.git
    cd UniData
    ```

2.  **克隆并编译 MMDetection3D v0.17.1**
    **注意**：请确保 `mmdetection3d` 文件夹位于 `UniData` 根目录下。

    ```bash
    # 在 UniData 目录下执行
    git clone https://github.com/open-mmlab/mmdetection3d.git
    cd mmdetection3d
    git checkout -f v0.17.1

    # 安装构建依赖（预防性措施，防止 setup.py 报错）
    pip install setuptools==60.2.0 

    # 从源码安装
    pip install -v -e .
    ```

### 第五步：安装 ASAP 数据处理依赖

要在 UniData 环境中运行插帧代码，需要稀疏卷积库 `spconv`。针对 CUDA 11.1 环境，请选择以下特定版本：

```bash
pip install spconv-cu111==2.1.25 cumm-cu111==0.2.9
```

### 第六步：解决关键依赖冲突（至关重要）

在配置过程中，可能会遇到由 `numpy` 版本引起的 `np.long` 报错，以及 `requests` 库的版本冲突。请务必执行以下修正方案。

#### 1. 修正 Numpy 版本 (解决 `AttributeError: module 'numpy' has no attribute 'long'`)

新版 Numpy (1.24+) 移除了 `np.long` 类型，导致旧版 MMLab 代码报错。此外，直接通过 pip 降级可能会破坏 Conda 底层的数学库链接 (mkl/blas)。

**解决方案**：使用 Conda 强制重装指定版本的 Numpy 和图像处理库。

```bash
# 此步骤不仅修正 numpy 版本，还能修复可能缺失的底层库依赖
conda install numpy=1.19.5 pillow imageio -y
```

#### 2. 修正 Requests 版本 (解决 OpenXLab 依赖冲突)

`openxlab` 需要较旧版本的 `requests`，而 Jupyter 组件通常需要较新版本。本项目中，优先保证代码运行环境的稳定性。

**解决方案**：降级 requests。

```bash
pip install requests==2.28.2
```

*注意：执行此命令后，如果看到关于 `jupyterlab-server` 依赖冲突的红色错误提示，请直接忽略。只要终端显示 `Successfully installed requests-2.28.2` 即视为成功。此冲突不会影响 UniData 和 ASAP 的核心功能运行。*

### 第七步：安装其他必要工具

安装 NuScenes 数据集开发工具包。

```bash
pip install nuscenes-devkit==1.1.9
```

### 第八步：验证环境

完成上述步骤后，使用 `pip list` 检查关键包的版本是否符合预期：

  * `torch`: 1.9.1+cu111
  * `mmcv-full`: 1.4.0
  * `mmdet`: 2.14.0
  * `mmsegmentation`: 0.14.1
  * `mmdetection3d`: 0.17.1 (路径应指向 UniData/mmdetection3d)
  * `numpy`: 1.19.5
  * `spconv-cu111`: 2.1.25

### 第九步：创建数据软链接

为了避免复制大量数据占用空间，建议在项目根目录下创建一个名为 `data` 的软链接，指向您实际的数据集存储目录。

1.  **确保您在项目根目录：**

    ```bash
    cd UniData
    ```

2.  **执行链接命令：**
    将 `<your_real_data_path>` 替换为您存放 NuScenes 等数据的真实绝对路径（例如 `/mnt/EC0A5E060A5DCE68/Ubuntu22_04/data` 或 `/data/nuscenes_root`）。

    ```bash
    # 语法: ln -s <源绝对路径> <目标链接名>
    ln -s <your_real_data_path> data
    ```

    **示例** (假设您的数据在 `/mnt/disk1/datasets`):

    ```bash
    ln -s /mnt/disk1/datasets data
    ```

3.  **验证链接是否成功：**
    运行以下命令。您应该看到 `data` 指向您的真实路径（由蓝色箭头 `->` 指示）：

    ```bash
    ls -l data
    ```

    *预期输出示例:* `lrwxrwxrwx 1 kevin kevin ... data -> /mnt/disk1/datasets`

    此时，您的逻辑目录结构应如下所示：

    ```text
    UniData/
    ├── mmdetection3d/
    ├── tools/
    ├── ...
    └── data/  (软链接) -> 指向您的真实数据盘
        ├── nuscenes/
        └── ...
    ```

### 第十步：下载预训练权重

我们需要下载 **CenterPoint Voxel 0.075 + DCN** 版本（mAP 56.92 / NDS 65.27），这是官方提供的最强模型。请在项目根目录下执行以下命令下载，或从 [CenterPoint 模型库](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/centerpoint/README.md) 手动下载。

```bash
# 1. 创建权重存放目录
mkdir -p assets/ckpts

# 2. 下载并重命名为 full_centerpoint.pth
wget https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth -O assets/ckpts/full_centerpoint.pth
```

至此，UniData 环境配置已全部完成，该环境包含了运行 ASAP 插帧处理模块所需的所有依赖。您现在可以继续进行预训练模型准备和后续的数据处理流程。