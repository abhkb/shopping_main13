# 环境配置说明

# 配置 conda
# 使用.condarc配置镜像源并将虚拟环境和安装的包转移到其它磁盘

# 安装 conda
conda --version

# 信息查看
conda info

# 创建环境
# 打开 pycharm 使用 conda 虚拟环境来配置 python 解释器

# 进入环境
conda activate shopping26

# 升级 pip 25.0.1
python -m pip install --upgrade pip

# 安装 opencv 4.13.0.92 自动包含 numpy 1.24.4
pip install opencv-python==4.13.0.92 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 opencv 4.10.0.84 用于 MSMF 后端框架
pip install opencv-contrib-python==4.10.0.84 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 验证安装
pip list

# 安装 dlib 19.24.6
conda install -c https://conda.anaconda.org/conda-forge dlib

# 安装 face_recognition 1.3.0
pip install face_recognition==1.3.0

# 验证安装
pip list

# 安装 yolov8 8.4.43
pip install ultralytics==8.4.43 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 pytorch cuda 11.8
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 566.24                 Driver Version: 566.24         CUDA Version: 12.7     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   39C    P0             12W /  125W |       0MiB /   8188MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+

# 验证安装
pip list

# 安装 paddlepaddle 3.0.0 paddleocr 3.5.0 paddlex 3.5.1
# 首次运行 paddleocr 会下载模型文件存放在 c盘 以后运行就无需下载
pip install paddlepaddle==3.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install paddleocr==3.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 pyserial
pip install pyserial -i https://pypi.tuna.tsinghua.edu.cn/simple

# 清理缓存
pip cache purge

# 查看缓存
pip cache info

# 运行代码前在 cmd 内输入以下代码 确保开启大模型
ollama run deepseek-r1:7b
