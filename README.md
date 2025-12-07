# UniData Environment Setup Guide

**Abstract**: This guide documents how to reproduce the UniData algorithm on a Linux environment (Ubuntu 22.04) and integrate the data processing module (frame interpolation) from the ASAP project. Given the strict version dependencies between OpenMMLab libraries (mmdet, mmseg, mmcv), CUDA, and PyTorch, and the need for this environment to support both the basic operation of VAD and the sparse convolution requirements of ASAP, please execute the following steps in strict order. **Do not upgrade core library versions arbitrarily.**

**Prerequisites**:

  * **OS**: Ubuntu 22.04/20.04 (Recommended)
  * **GPU Driver**: Version compatible with CUDA 11.1
  * **Anaconda / Miniconda**: Installed

-----

### Step 1: Create and Activate Virtual Environment

To ensure low-level compatibility, use Python 3.8.

```bash
conda create -n unidata python=3.8 -y
conda activate unidata
```

### Step 2: Install PyTorch and CUDA Runtime

Install PyTorch 1.9.1 with CUDA 11.1.

```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### Step 3: Install OpenMMLab Basic Dependencies

The UniData project strictly locks the versions of `mmcv-full`, `mmdet`, and `mmsegmentation`. Before installing specific libraries, install the `openmim` management tool, although we will specify versions manually to ensure precision.

1.  **Install OpenMIM**

    ```bash
    pip install openmim
    ```

2.  **Install MMCV-Full**
    **Note**: Must be version 1.4.0.

    ```bash
    pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.1/index.html
    ```

3.  **Install MMDetection and MMSegmentation**
    **Strictly forbid** upgrading these two libraries, otherwise API calls in the UniData code will fail.

    ```bash
    pip install mmdet==2.14.0
    pip install mmsegmentation==0.14.1
    ```

4.  **Install Timm**

    ```bash
    pip install timm
    ```

### Step 4: Compile and Install MMDetection3D

To facilitate debugging and integration, we will compile and install `mmdetection3d` from source within the UniData project directory.

1.  **Clone UniData repository (if not already cloned) and prepare directory**

    ```bash
    # Assuming you are in the workspace root
    git clone https://github.com/hustvl/UniData.git
    cd UniData
    ```

2.  **Clone and Compile MMDetection3D v0.17.1**
    **Note**: Ensure the `mmdetection3d` directory is located inside the `UniData` root.

    ```bash
    # Execute inside the UniData directory
    git clone https://github.com/open-mmlab/mmdetection3d.git
    cd mmdetection3d
    git checkout -f v0.17.1

    # Install build dependencies (preventative measure)
    pip install setuptools==60.2.0 

    # Install from source
    pip install -v -e .
    ```

### Step 5: Install ASAP Data Processing Dependencies

To run interpolation code in the UniData environment, the sparse convolution library `spconv` is required. For the CUDA 11.1 environment, select the specific version below:

```bash
pip install spconv-cu111==2.1.25 cumm-cu111==0.2.9
```

### Step 6: Resolve Key Dependency Conflicts (Crucial)

During configuration, you may encounter `np.long` errors caused by `numpy` versions and version conflicts with the `requests` library. Please correct them using the following solutions.

#### 1\. Fix Numpy Version (Solves `AttributeError: module 'numpy' has no attribute 'long'`)

Newer versions of Numpy (1.24+) removed the `np.long` type, causing errors in older MMLab code. Additionally, directly downgrading via pip might break underlying Conda math library links (mkl/blas).

**Solution**: Use Conda to forcibly reinstall the specified Numpy version and image libraries.

```bash
# This step fixes the numpy version and restores potential missing low-level libraries
conda install numpy=1.19.5 pillow imageio -y
```

#### 2\. Fix Requests Version (Solves OpenXLab Dependency Conflict)

`openxlab` requires an older version of `requests`, while Jupyter components often need newer ones. In this project, prioritize the code execution environment.

**Solution**: Downgrade requests.

```bash
pip install requests==2.28.2
```

*Note: After executing this command, if you see red error messages regarding `jupyterlab-server` dependency conflicts, please ignore them. As long as the terminal shows `Successfully installed requests-2.28.2`, it is considered successful. This conflict does not affect the core operation of UniData and ASAP.*

### Step 7: Install Other Necessary Tools

Install the NuScenes dataset development toolkit.

```bash
pip install nuscenes-devkit==1.1.9
```

### Step 8: Verify Environment

After completing the steps above, use `pip list` to check if key package versions meet expectations:

  * `torch`: 1.9.1+cu111
  * `mmcv-full`: 1.4.0
  * `mmdet`: 2.14.0
  * `mmsegmentation`: 0.14.1
  * `mmdetection3d`: 0.17.1 (Path should point to UniData/mmdetection3d)
  * `numpy`: 1.19.5
  * `spconv-cu111`: 2.1.25

### Step 9: Create Data Symlink

To avoid duplicating large amounts of data and consuming space, create a symbolic link named `data` in the project root directory pointing to your actual dataset directory.

1.  **Ensure you are in the project root:**

    ```bash
    cd UniData
    ```

2.  **Execute the link command:**
    Replace `<your_real_data_path>` with the absolute path where you actually store NuScenes and other data (e.g., `/mnt/EC0A5E060A5DCE68/Ubuntu22_04/data` or `/data/nuscenes_root`).

    ```bash
    # Syntax: ln -s <source_absolute_path> <target_link_name>
    ln -s <your_real_data_path> data
    ```

    **Example** (Assuming your data is under `/mnt/disk1/datasets`):

    ```bash
    ln -s /mnt/disk1/datasets data
    ```

3.  **Verify if the link is successful:**
    Run the following command. You should see `data` pointing to your real path (indicated by a blue arrow `->`):

    ```bash
    ls -l data
    ```

    *Expected output example:* `lrwxrwxrwx 1 kevin kevin ... data -> /mnt/disk1/datasets`

    At this point, your logical directory structure should look like this:

    ```text
    UniData/
    ├── mmdetection3d/
    ├── tools/
    ├── ...
    └── data/  (Symlink) -> Points to your actual data drive
        ├── nuscenes/
        └── ...
    ```

At this point, the UniData environment configuration is complete and contains all dependencies required to run the data processing modules. You can now proceed with pre-trained model preparation and data processing workflows.