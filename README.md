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

## 核心文件

- `scripts/prepare_manifest.py`：从 `labels.xlsx` 生成含提示词的训练清单
- `scripts/train_text_to_image.py`：训练文本条件 DDPM 生成模型
- `scripts/generate_images.py`：给定提示词生成 DOBI NIR 图像
- `scripts/evaluate_generation.py`：生成图像质量与多样性评估
- `scripts/train_classifier.py`：真实/合成增强分类验证
- `docs/research_design.md`：完整毕业设计方案

## 实验逻辑

1. 用医学结构化字段构造英文提示词，兼容 CLIP 文本编码器。
2. 训练小分辨率文本条件 DDPM，重点保留 DOBI 灰度成像形态。
3. 用特征 FID/KID、最近邻相似度和强度统计评估生成质量。
4. 用真实训练集与“真实+合成”训练集分别训练分类器，在固定测试集上比较 ROC-AUC、PR-AUC、灵敏度、特异度和 F1。

注意：该项目输出仅用于科研实验，不能作为临床诊断依据。
