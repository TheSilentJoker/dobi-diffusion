# BreastDiffusion-ai

面向 DOBI 近红外乳腺图像的文本生成图像与肿瘤二分类毕业设计实验框架。

本项目使用 `data/labels.xlsx` 中的结构化医学标签生成文本提示词，将 `data/processed` 中的 DOBI NIR 灰度图像组织为文本到图像训练集；随后训练文本条件扩散模型生成类似 NIR 图像，并用下游肿瘤有无分类任务验证合成图像的医学有效性。

## 数据

- 图像：`data/processed/*.bmp`
- 标签：`data/labels.xlsx`
- 标签含义：`label=1` 表示有肿瘤图像，`label=0` 表示无肿瘤图像
- 已有划分：`split=train/val/test`

## 快速开始

```powershell
python -m pip install -r requirements.txt
python scripts/prepare_manifest.py
python scripts/train_classifier.py
```

训练文本条件扩散模型需要 PyTorch、Diffusers 和 GPU：

```powershell
python scripts/train_text_to_image.py --manifest data/manifests/dobi_prompts.csv --output-dir outputs/text_ddpm --epochs 100 --batch-size 16
python scripts/generate_images.py --model-dir outputs/text_ddpm --label 1 --num-prompts 25 --num-per-prompt 8 --output-dir outputs/generated_tumor
python scripts/evaluate_generation.py --generated-manifest outputs/generated_tumor/generated_manifest.csv
python scripts/train_classifier.py --synthetic-manifest outputs/generated_tumor/generated_manifest.csv --output reports/classification_with_synthetic.json
```

使用 SwanLab 实时记录训练曲线：

```powershell
python scripts/train_text_to_image.py --swanlab --swanlab-project BreastDiffusion-ai --swanlab-experiment text-ddpm-full-prompt --log-every 10
```

SwanLab 只记录训练曲线核心指标：`step/loss`、`step/noise_loss`、`step/background_black_loss`、`step/lr` 使用 global step 作为横坐标；`epoch/loss`、`epoch/noise_loss`、`epoch/background_black_loss`、`epoch/lr` 使用 epoch 作为横坐标。如果不想同步到云端，可以加 `--swanlab-mode local` 或 `--swanlab-mode offline`。

训练脚本默认会在 `--output-dir` 下按运行时间创建独立实验目录，例如 `outputs/text_ddpm/20260521_143022_exp01`，并在该目录内按 `--save-every` 保存 checkpoint，例如 `checkpoint-epoch-010`，同时更新该 run 内部的 `latest`。生成脚本传入实验根目录时会自动读取最新 run 的 `latest`；也可以显式指定某个 checkpoint：

```powershell
python scripts/generate_images.py --model-dir outputs/text_ddpm/20260521_143022_exp01/checkpoint-epoch-010 --label 1 --output-dir outputs/generated_epoch010
```

`generate_images.py` 默认会在 `--output-dir` 下按运行时间创建子目录，例如 `outputs/custom_prompt_preview/20260517_213045_exp01`，便于区分不同实验。若需要直接写入指定目录，可加 `--no-run-subdir`。

## 核心文件

- `scripts/prepare_manifest.py`：从 `labels.xlsx` 生成含提示词的训练清单
- `scripts/train_text_to_image.py`：训练文本条件 DDPM 生成模型
- `scripts/generate_images.py`：给定提示词生成 DOBI NIR 图像
- `scripts/evaluate_generation.py`：生成图像质量与多样性评估
- `scripts/train_classifier.py`：真实/合成增强分类验证
- `docs/research_design.md`：完整毕业设计方案

## 实验逻辑

1. 用医学结构化字段构造英文提示词，兼容 CLIP 文本编码器。
2. 训练 DOBI 专用文本条件 DDPM，使用前景 mask、数值成像指标、背景约束、Min-SNR 和 EMA 保留 DOBI 灰度成像形态。
3. 用特征 FID/KID、最近邻相似度、黑背景比例、前景比例和强度统计评估生成质量。
4. 用真实训练集与“真实+合成”训练集分别训练分类器，在固定测试集上比较 ROC-AUC、PR-AUC、灵敏度、特异度和 F1。

注意：该项目输出仅用于科研实验，不能作为临床诊断依据。

## DOBI 优化模型

推荐主方法：

```powershell
python scripts/train_text_to_image.py --manifest data/manifests/dobi_prompts.csv --output-dir outputs/dobi_mask_metadata_ddpm --condition-mode text_mask_metadata --epochs 30 --batch-size 16 --save-every 10 --min-snr-gamma 5 --background-black-loss-weight 0.05 --swanlab --swanlab-experiment dobi-mask-metadata
python scripts/generate_images.py --model-dir outputs/dobi_mask_metadata_ddpm --label 1 --num-prompts 25 --num-per-prompt 4 --freeu --output-dir outputs/dobi_mask_metadata_samples --run-name tumor
python scripts/evaluate_generation.py --generated-manifest outputs/dobi_mask_metadata_samples/<run_dir>/generated_manifest.csv --output reports/dobi_mask_metadata_generation.json
```

关键创新模块：

- DOBI foreground mask condition：显式约束黑背景和半椭圆乳腺前景。
- Metadata conditioner：把 `Breast_Ratio_Global(%)`、`Illum_Coverage_Local(%)`、`Leak_Ratio_Local(%)` 编码成额外条件 token。
- DOBI weighted loss：前景/背景分区加权，并加入背景压黑约束。
- Min-SNR + EMA：提升小数据扩散训练稳定性。
- FreeU sampling：采样时增强 U-Net backbone/skip 特征平衡。

生成质量不理想时，优先比较 `generated_dark_background_ratio`、`generated_foreground_ratio` 与真实图的差值。

## 消融实验

生成实验计划但不执行：

```powershell
python scripts/run_ablation_experiments.py --epochs 30 --batch-size 16
```

执行完整消融：

```powershell
python scripts/run_ablation_experiments.py --execute --stage all --epochs 30 --batch-size 16 --swanlab
```

默认包含：

- `text_only`：原始文本条件 DDPM
- `text_mask`：加入 DOBI mask 与背景约束
- `text_mask_metadata`：主方法
- `no_metadata`：去掉数值成像指标
- `no_min_snr`：去掉 Min-SNR
- `no_bg_loss`：去掉背景压黑约束
