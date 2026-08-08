# GAME 项目工作指南

## 运行环境（重要）

本项目依赖独立 conda 虚拟环境运行，不要在 base 环境或系统 Python 中直接执行脚本。

- **环境名**：`game`（Python 3.12）
- **激活方式**：`conda activate game`
- **PyTorch**：2.11.0+cu128（CUDA 12.8，手动用 pip 安装，未列入 requirements.txt）

开始任何工作前，先在该环境内运行：

```bash
conda activate game
cd /opt/work/open-source-project/GAME
```

## 推理运行

```bash
conda activate game
python infer.py extract [path-or-directory] -m [model-path] --glob *.wav --output-formats mid,txt,csv
```

- `infer.py` 的 `-m` 指向 `.pt` 模型权重；模型同目录需有 `config.yaml`（`load_inference_model` 会读取），开启语言支持时还需 `lang_map.json`。
- 预训练模型下载：https://github.com/openvpi/GAME/releases

## 依赖说明

`requirements.txt` 覆盖了大部分依赖，但有**两个推理必需包未列出**（已在 `game` 环境内补装）：

- `praat-parselmouth`（`lib/feature/pitch.py` 中 `import parselmouth`，提供 Praat 的 `Sound`/`to_pitch_ac` 接口）
- `pyworld`（同文件，`pyworld.harvest` 等）

> 注意：PyPI 上有个同名旧包 `parselmouth-1.1.1`（Google Ads，无关），不要误装；正确包名是 `praat-parselmouth`。

只推理、不训练时，理论上可省 `sympy`、`onnx`/`onnxscript`/`onnxslim`、`tensorboard`/`tensorboardX`，但因 import 链耦合（`inference/data.py` → `training/data.py` → `training/augmentation.py`），`colorednoise`、`scipy` 实为推理必需，不可省。最省心做法仍是 `pip install -r requirements.txt` 全装。

### ONNX 推理（可选，需额外依赖）

仓库自带 `infer.py` 只支持 PyTorch `.pt` 模型。若手头只有 ONNX 格式模型（`onnx/GAME-*` 目录下的 `encoder/segmenter/estimator/dur2bd/bd2dur.onnx` + `config.json`），用项目根目录的 `infer_onnx.py` 脚本，不依赖 Lightning/PyTorch：

```bash
conda activate game
python infer_onnx.py inputFiles/sws_vocals.wav -m onnx/GAME-1.0.3-medium-onnx -l zh --output-formats mid,txt
```

- 参数与 `infer.py` 对齐（`--seg-threshold`/`--seg-radius`/`--est-threshold`/`--t0`/`--nsteps`/`--tempo`）。
- `--cpu` 强制 CPU；默认自动尝试 CUDA，不可用则回退 CPU。

**onnxruntime 版本约束（重要）**：必须装 `onnxruntime-gpu==1.20.2`，**不要升级到 1.23+**。

- 1.20.x 对应 CUDA 12.x + cuDNN 9，复用 PyTorch cu128 已装的 CUDA 12.8 / cuDNN 9 库（`libcublasLt.so.12`），CUDA provider 可正常加载。
- 1.23+（尤其 1.28）需要 CUDA 13 + cuDNN 9，会找 `libcublasLt.so.13`，与当前 PyTorch cu128 冲突，CUDA provider 加载失败、回退 CPU 并刷大量警告。
- 装 1.20.2 用 `pip install onnxruntime-gpu==1.20.2`；若用 `--force-reinstall` 会误把 numpy 升到 2.x，需随后 `pip install numpy==1.26.4` 钉回（项目要求 `numpy<2.0.0`）。

> 性能：单音频逐段处理时 CPU 已够快（~37s 跑 218s 音频，约 6× 实时），GPU 因模型小/单段计算量小+数据拷贝开销反而略慢；批量处理长音频时 GPU 才有优势。

## Git 约束

未经用户同意，不要执行 git commit / push，也不要新建分支开发。
