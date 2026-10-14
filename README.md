# YOLOv8 / SFCHD 训练代码快照

此仓库保存当前本地修改后的训练代码，供后续迁移到 Kaggle GPU 使用。

## 本次上传范围

- `training-code.zip`：完整 `ultralytics/` 运行时、`pyproject.toml`、原项目 `README.md` 和 `LICENSE`。
- `SOURCE_MANIFEST.json`：源提交号、工作区状态、压缩包和每个源码文件的 SHA-256。
- 包内包含 `ultralytics/cfg/datasets/SFCHD_paper.yaml`，保留当前本地版本。

代码来自本地工作目录的实际文件，保留已有修改。未上传 `.git` 历史、YOLOv5、文档树、示例、测试、缓存、训练结果、数据集或模型权重。核心包内部的小型资源文件保留，以保证正常导入和运行。

这次先以压缩包上传源码快照，避免网页上传大量目录文件。以后需要逐行查看差异或持续开发时，可将包内源码解压后用 Git 提交为正常目录结构。

## 在 Kaggle 中使用

本地磁盘不会自动挂载到 Kaggle。训练代码、图片/标签、输出分别管理：GitHub 保存代码快照，Kaggle 私有 Dataset 提供输入，`/kaggle/working` 保存输出。

这个 GitHub 仓库为公开仓库，因此 Kaggle 可以直接拉取，无需 GitHub 密码或令牌。在已开启 Internet 的 Notebook 中依次运行：

```bash
!git clone https://github.com/jiaweipan176/yolo-sfchd-train.git /kaggle/working/yolo-repo
!git -C /kaggle/working/yolo-repo rev-parse HEAD
```

```python
from pathlib import Path
from zipfile import ZipFile
import subprocess
import sys

source = Path('/kaggle/working/yolo-source')
source.mkdir(parents=True, exist_ok=True)
with ZipFile('/kaggle/working/yolo-repo/training-code.zip') as package:
    package.extractall(source)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', str(source)], check=True)
```

新会话可重新运行这些单元格；同一会话不要重复 clone 到已存在目录。记录 `rev-parse HEAD` 的提交号，在后续正式实验中固定该提交，以免代码更新改变实验环境。

如果网络无法访问 GitHub，也可以使用下面的文件上传路线：

1. 从本仓库下载 `training-code.zip`，作为 Kaggle **私有 Dataset** 上传。
2. 建立 Kaggle Notebook，使用 **Add Input / Add Data** 添加这个代码 Dataset。
3. 先保持 Accelerator 为 None，确认 Internet 已开启，安装并检查源码。开始训练时再选择 **GPU T4 x2**，用 `nvidia-smi` 检查实际设备和平台剩余配额。
4. 将下列 Python 代码放入 Notebook 单元格运行：

```python
from pathlib import Path
from zipfile import ZipFile
import subprocess
import sys

archives = list(Path('/kaggle/input').rglob('training-code.zip'))
assert len(archives) == 1, '请添加一个包含 training-code.zip 的代码 Dataset'
source = Path('/kaggle/working/yolo-source')
source.mkdir(parents=True, exist_ok=True)
with ZipFile(archives[0]) as package:
    package.extractall(source)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', str(source)], check=True)
print('源码目录：', source)
```

安装后如已在当前内核导入过其他版本的 `ultralytics`，重启内核，然后检查：

```python
import ultralytics
import torch
print('源码：', ultralytics.__file__)
print('框架版本：', ultralytics.__version__)
print('PyTorch：', torch.__version__)
print('CUDA 可用：', torch.cuda.is_available())
print('GPU 数量：', torch.cuda.device_count())
assert '/kaggle/working/yolo-source/' in ultralytics.__file__
```

必须从本次上传的源码安装，才能使用本地修改。直接 `pip install ultralytics` 安装的是发布包。

## 数据就绪之后才启动训练

图片和标签尚未上传。包内 SFCHD YAML 与现有图片清单还含 `D:/Model/yolo/...` 本机路径，不能直接作为 Kaggle 数据配置使用。后续需单独上传 `images/`、`labels/`、原划分清单，并在云端生成新 YAML 和图片清单，保留原来的划分成员。

论文口径的当前配置为 7 类，train 9897 张、val 2475 张；val 对应论文的 Testing，当前这套划分没有独立 test。不要在迁移时随机重新划分数据，也不要将后续的小样本实验与完整复现实验混为一谈。

数据配置生成后，先运行 1 个 epoch 检查数据读取和权重保存。例如在 Notebook 代码单元格中运行：

```bash
!yolo detect train model=yolov8n.pt data=/kaggle/working/SFCHD_kaggle.yaml epochs=1 imgsz=640 batch=16 device=0 workers=2 project=/kaggle/working/runs name=sfchd_smoke
```

双 GPU 可设置 `device=0,1`，batch 是总批量，需可被 2 整除；两张卡的显存不会自动合成一张卡的显存。正式实验的模型型号、epoch、优化器、学习率、增强和预训练方式必须另行对照论文。上面的短训练参数只用于检查运行链路。

长训练使用 **Save Version → Save & Run All** 在后台运行。会重新从头执行 Notebook，应避免同时保留一个正在训练的交互会话。保存或下载 `best.pt`、可恢复的中间检查点、`results.csv`、`args.yaml` 和环境信息；不要假定临时运行目录自动持久保存。每周配额与单次会话时限以当前账号页面为准。

## 官方参考

- [Kaggle Notebook 使用说明](https://www.kaggle.com/docs/notebooks)
- [Kaggle GPU 使用与配额](https://www.kaggle.com/docs/efficient-gpu-usage)
- [Ultralytics 训练说明](https://docs.ultralytics.com/modes/train/)

本仓库上传完成仅代表训练代码已保存；尚未进行云端训练或验证论文指标。
