# YOLOv8 / SFCHD 训练源码

本仓库直接保存当前修改后的 Ultralytics 训练源码，可在 VS Code 中提交、推送，并由 Kaggle 拉取运行。源码已从压缩包迁移为普通文件目录。

## 本地开发目录

```text
D:\Model\yolo\kaggle\yolo-sfchd-train
├── ultralytics/            # 训练框架源码、配置及运行资源
├── pyproject.toml          # 安装与依赖配置
├── LICENSE                 # 原项目许可证
├── README.upstream.md      # 原项目说明
├── SOURCE_ORIGIN.md        # 初始迁移来源
└── .gitignore              # 排除数据集、权重、结果和缓存
```

以后在这个目录修改要同步到 GitHub 的训练代码。原来的 `D:\Model\yolo\yolov8` 保留，不与此目录自动同步；在旧目录修改代码，不会出现在本仓库的待提交列表里。

## 用 VS Code 提交和推送

1. VS Code → **文件 → 打开文件夹**，选择上面的开发目录；也可打开旁边的 `yolo-sfchd-train.code-workspace`。
2. 修改源码并保存。
3. 左侧 **源代码管理**，查看修改差异，点击文件旁的 **+** 暂存。
4. 输入本次改动的说明，点击 **提交**。
5. 点击 **推送 / 同步更改**；首次需要时按 GitHub 登录提示操作。
6. GitHub 页面查看最新提交，确认更新成功。

保存文件只保存到本机；提交记录版本；推送后 GitHub 才更新。执行“同步更改”前应查看是否存在需要拉取的远程提交。

如希望在本机使用这份源码，先激活自己的 YOLO Python 环境，在这个目录的终端执行：

```bash
python -m pip install -e .
python -c "import ultralytics; print(ultralytics.__file__)"
```

确认打印路径位于本目录。此操作只在你指定的 Python 环境中执行，不要直接在未知的系统环境里安装。

## Kaggle 拉取与安装

在 Kaggle 创建 Notebook，开启 **Internet**。先使用 CPU 检查环境，开始训练时再选择 **GPU T4 x2**。

在一个代码单元格中运行：

```python
from pathlib import Path
import subprocess
import sys

repo = Path('/kaggle/working/yolo-sfchd-train')
subprocess.run([
    'git', 'clone', 'https://github.com/jiaweipan176/yolo-sfchd-train.git', str(repo)
], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', str(repo)], check=True)
subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], check=True)
```

新会话从头拉取；同一会话不要重复 clone 到已有目录。代码更新后，在**没有训练任务运行**时，拉取更新：

```python
subprocess.run(['git', '-C', str(repo), 'pull', '--ff-only'], check=True)
```

随后重启 Notebook 内核，再重新导入框架。若依赖配置也更新了，需要重新执行源码安装。避免让正在训练的任务读到途中变更的代码。

重启后检查来源与 GPU：

```python
import ultralytics
import torch
print('源码：', ultralytics.__file__)
print('框架版本：', ultralytics.__version__)
print('PyTorch：', torch.__version__)
print('CUDA 可用：', torch.cuda.is_available())
print('GPU 数量：', torch.cuda.device_count())
assert '/kaggle/working/yolo-sfchd-train/' in ultralytics.__file__
```

正式实验保存 Git 提交号、依赖版本和数据版本。复跑既有实验时，在安装前执行 `git checkout <记录的完整提交号>`，不直接使用可能已经变化的最新代码。

## 数据与训练输出

训练图片、标签、预训练权重和运行结果不纳入本仓库。数据单独上传到 Kaggle **私有 Dataset**，在 Notebook 用 **Add Input / Add Data** 添加；输出写入 `/kaggle/working`。

`ultralytics/cfg/datasets/SFCHD_paper.yaml` 保留当前本地配置，其中的数据根路径仍为 `D:/Model/yolo/...`。迁移训练数据时，必须在云端生成新的 YAML 和图片清单，不可只改 YAML 而保留清单中的 Windows 绝对路径。

当前论文口径配置为 7 类，train 9897 张、val 2475 张；val 对应论文的 Testing，没有独立 test。迁移时保留原划分成员，不重新随机划分。完整复现实验与小样本预标注实验分别记录。

数据配置准备好后，先做 1 个 epoch 的运行检查：

```bash
!yolo detect train model=yolov8n.pt data=/kaggle/working/SFCHD_kaggle.yaml epochs=1 imgsz=640 batch=16 device=0 workers=2 project=/kaggle/working/runs name=sfchd_smoke
```

正式实验的型号、预训练方式、epoch、优化器、学习率和增强设置另行对照论文。双 GPU 设置 `device=0,1`，batch 是总批量，应能被 2 整除。上述短训练参数只检查链路，不是论文复现参数。

长训练使用 **Save Version → Save & Run All**，会从头执行 Notebook；避免同时保留正在训练的交互会话。保存或下载 `best.pt`、可恢复的中间检查点、`results.csv`、`args.yaml` 和环境信息。配额与会话时限以当前账号页面为准。

## 官方参考

- [Kaggle Notebook 使用说明](https://www.kaggle.com/docs/notebooks)
- [Kaggle GPU 使用与配额](https://www.kaggle.com/docs/efficient-gpu-usage)
- [Ultralytics 训练说明](https://docs.ultralytics.com/modes/train/)

源码上传与 Git 关联完成，不代表云端训练已执行或论文指标已验证。
