# 医学领域文本生成图像技术在 DOBI 近红外乳腺图像中的研究设计

## 1. 课题目标

本课题面向 DOBI 近红外乳腺图像，研究如何利用结构化医学标签构造文本提示词，并训练文本条件图像生成模型，使模型能够根据“有无肿瘤、左右侧、BI-RADS、病灶形态、边界、内部回声、钙化、血流、光照覆盖率”等描述生成近似真实分布的 NIR 图像。生成图像质量达到预期后，将其用于下游肿瘤二分类任务，验证合成数据是否能提升小样本正类识别能力。

## 2. 数据分析结论

当前数据包含 4275 张 DOBI NIR 图像，均为 `128x102` 灰度 BMP。`labels.xlsx` 共 28 个字段，其中 `filename` 与图像文件一一对应，`label=1` 表示有肿瘤，`label=0` 表示无肿瘤。类别极度不平衡：负类 4233 张，正类 42 张。已有划分如下：

| split | label=0 | label=1 |
|---|---:|---:|
| train | 2540 | 25 |
| val | 847 | 8 |
| test | 846 | 9 |

实验必须固定该划分，尤其不能把同一图像或由测试提示词生成的图像混入训练阶段。

## 3. 文本提示词设计

提示词采用英文，原因是默认 CLIP 文本编码器对英文医学描述更稳定。每条提示词由四类信息组成：

1. 成像模态：`near-infrared dynamic optical breast imaging, DOBI NIR grayscale medical image, low-resolution 128 by 102 pixels`
2. 解剖与标签：左/右乳腺、`tumor-positive` 或 `tumor-negative`
3. 医学结构化属性：BI-RADS、单发/多发、位置、最大径、形态、纵横比、边界、边缘、内部回声、后方回声、钙化、血流
4. DOBI 质量与强度属性：全局乳腺占比、局部光照覆盖率、局部漏光比例

正类额外加入 `subtle asymmetric vascular and absorption pattern`，负类加入 `regular symmetric illumination pattern without tumor signature`。负向提示词排除彩色照片、X 光、CT、MRI、超声截图、文字水印和严重伪影。

提示词由 `scripts/prepare_manifest.py` 自动生成，输出 `data/manifests/dobi_prompts.csv`。

## 4. 生成模型选择

默认方案选择文本条件像素空间 DDPM，而不是直接从零训练大型 Stable Diffusion。理由如下：

- 图像分辨率低且统一，像素空间扩散足够覆盖主要形态。
- 医学数据量小，尤其正类只有 42 张，直接微调大模型容易过拟合并生成 RGB/自然图像偏差。
- DDPM 可以使用冻结 CLIP 文本编码器保留文本条件能力，同时训练小型 UNet 适配 NIR 灰度分布。
- 训练、采样和评估流程更容易写入论文并复现。

模型输入为三通道灰度复制图，训练前将 `128x102` 图像居中补边到 `128x128`。采样后再裁剪回 `128x102`。训练使用类别均衡采样、提示词 dropout、AdamW 和余弦 beta DDPM scheduler。后续有更大数据和 GPU 时，可将该方案升级为 Stable Diffusion LoRA：保留同一份 prompt manifest，把灰度图复制为 RGB 后进行低秩微调。

## 5. 生成图像评估流程

生成质量不只看视觉效果，而是按“真实性、多样性、标签一致性、隐私风险、下游有效性”五个角度评估：

- 特征 FID：使用 DOBI 灰度像素、直方图、梯度和统计特征经 PCA 后计算 Fréchet distance。
- 特征 KID：使用三阶多项式 MMD 衡量真实/合成分布差异。
- 强度统计：比较真实和合成图像的均值、方差、梯度强度和灰度直方图。
- 最近邻相似度：计算每张合成图与训练集中最近真实图的余弦相似度，过高提示记忆化风险。
- 人工质控：检查是否保留 DOBI 半椭圆乳腺区域、中心光照/吸收纹理、灰度边界，是否出现文字、彩色、棋盘伪影。

建议阈值：特征 FID/KID 相对初始模型持续下降；最近邻最大相似度不能长期接近 1；正负类合成图应在视觉上存在可解释差异，但不能靠明显伪影区分类别。

## 6. 下游分类验证

下游任务为给定 NIR 图像判断有无肿瘤。实验分三组：

1. 真实数据基线：只用真实训练集训练分类器。
2. 合成增强：真实训练集 + 生成正类图像训练分类器。
3. 消融实验：不同生成数量、不同 guidance scale、不同提示词字段组合。

固定真实验证集调阈值，固定真实测试集报告结果。核心指标包括 ROC-AUC、PR-AUC、balanced accuracy、sensitivity、specificity、precision、F1 和混淆矩阵。由于正类稀少，PR-AUC、sensitivity 和 specificity 比普通 accuracy 更重要。

当前仓库提供 `scripts/train_classifier.py` 作为可复现基线：灰度像素、直方图、梯度统计特征 + PCA + 类别加权 Logistic Regression。后续可增加 ResNet18/EfficientNet 小模型，但必须保持同一 train/val/test 划分。

## 7. 论文实验章节建议

1. 数据集与预处理：说明 DOBI 模态、灰度图尺寸、标签字段、类别不平衡和划分策略。
2. 提示词工程：展示字段到自然语言的映射，给出正负样例。
3. 模型方法：介绍文本编码器、条件 UNet、扩散前向/反向过程、类别均衡训练。
4. 生成质量评估：报告 FID/KID、最近邻、可视化样例和人工质控。
5. 下游任务验证：报告真实基线与合成增强对比，分析正类召回率变化。
6. 消融与讨论：提示词字段、合成数量、guidance scale、类别均衡采样的影响。
7. 局限性：正类样本过少、标签噪声、无病灶分割标注、生成图像不能替代真实临床样本。

## 8. 复现实验命令

```powershell
python scripts/prepare_manifest.py
python scripts/train_classifier.py
python scripts/train_text_to_image.py --epochs 100 --batch-size 16 --output-dir outputs/text_ddpm
python scripts/generate_images.py --model-dir outputs/text_ddpm --label 1 --num-prompts 25 --num-per-prompt 8 --output-dir outputs/generated_tumor
python scripts/evaluate_generation.py --generated-manifest outputs/generated_tumor/generated_manifest.csv
python scripts/train_classifier.py --synthetic-manifest outputs/generated_tumor/generated_manifest.csv --output reports/classification_with_synthetic.json
```

最终结论应以固定真实测试集的下游分类表现为准；只有当合成图像质量指标合理且分类灵敏度/PR-AUC 有稳定收益时，才能认为文本生成图像技术对该医学任务有实际帮助。
