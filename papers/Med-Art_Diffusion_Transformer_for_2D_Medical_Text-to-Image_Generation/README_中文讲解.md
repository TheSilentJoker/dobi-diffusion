# Med-Art: Diffusion Transformer for 2D Medical Text-to-Image Generation

## 论文信息

- 题目：Med-Art: Diffusion Transformer for 2D Medical Text-to-Image Generation
- 作者：Changlu Guo, Anders Nymark Christensen, Morten Rieger Hannemose
- 方向：医学图像文生图、扩散模型、Diffusion Transformer、医学数据增强
- arXiv：2506.20449
- arXiv 提交时间：2025-06-25
- DOI：10.48550/arXiv.2506.20449
- 项目页：https://medart-ai.github.io/
- 会议/出版信息：DGM4MICCAI 2025 Long Oral；Springer LNCS 章节页显示 First Online: 2025-09-25

## 这篇论文解决什么问题

通用文生图模型通常依赖大规模图文数据训练，但医学图像领域有两个典型困难：

1. 医学图像数据规模小，且受隐私、伦理、授权限制，不容易公开收集。
2. 医学图像通常缺少适合文生图模型训练的自然语言描述；即使有文本，也常是高度专业的病历或放射报告，和自然图像文生图预训练时见到的文本分布不一致。

这篇论文提出 Med-Art，目标是在有限医学数据上，把已有的通用文生图模型迁移到 2D 医学图像生成任务中，让模型能根据文本描述生成更真实的胃肠镜、皮肤镜等医学图像。

## 核心方法

论文的核心设计可以概括为三部分。

第一，使用视觉语言模型生成医学图像描述。作者提出 Visual Symptom Generator（VSG），用 LLaVA-Next 等视觉语言模型根据原始医学图像和类别信息生成更细致的自然语言描述。例如不只写“黑色素瘤皮肤镜图像”，而是描述颜色、边界、形状、纹理、皮肤镜模式等视觉特征。这样可以缓解医学图像缺少配套文本的问题。

第二，基于 PixArt-alpha 这类 Diffusion Transformer 文生图模型做参数高效微调。作者没有从零训练医学文生图模型，而是在预训练文生图模型上加入 LoRA，并且不仅微调 DiT 去噪网络，也对文本编码器做 LoRA，使文本侧更适应医学术语和医学视觉描述。

第三，提出 Hybrid-Level Diffusion Fine-tuning（HLDF）来处理颜色失真。普通 latent-space 微调容易在内镜、皮肤镜这类颜色敏感图像上产生过饱和颜色。HLDF 在训练中周期性采样生成图像，并在像素空间约束生成图和真实图在各颜色通道上的均值与标准差差异，从而让颜色分布更接近医学图像真实数据。

## 实验设置与结果

论文在两个 2D 医学图像数据集上评估：

- Kvasir：胃肠内镜图像，8 个类别，每类 1000 张。
- Skin lesions：皮肤病变/皮肤镜图像，包含 melanoma、nevus、seborrheic keratosis 等类别。

对比方法包括 Stable Diffusion 1.4、Stable Diffusion 1.5、SDXL、Hunyuan-DiT、PixArt-alpha、Fast-DiT 等。评价指标包括 FID、KID，以及作者为医学数据集替换特征提取器后的 KFD/HFD，还包含下游分类性能评估。

主要结论是：Med-Art 在两个数据集上整体优于对比方法。论文表格中，Med-Art 在 Kvasir 上的 FID 为 51.99，优于 PixArt-alpha 的 72.86；在 Skin lesions 上的 FID 为 67.45，优于 PixArt-alpha 的 73.68。作者还通过消融实验说明，VSG 生成的详细文本、文本编码器 LoRA、HLDF 像素级颜色约束都对结果有贡献。

## 对毕设/医学文生图研究的参考价值

这篇论文适合作为“医学领域文生图”方向的近期代表工作，因为它没有只停留在通用扩散模型套用医学数据，而是专门处理了医学文生图里的两个实际痛点：文本标注稀缺和颜色真实性。

如果你的课题和乳腺影像生成、医学数据增强或小样本医学图像生成有关，可以重点借鉴：

- 用 VLM 自动生成医学图像的细粒度描述，替代人工文本标注。
- 使用 LoRA 微调预训练文生图模型，降低训练成本。
- 在医学图像中加入颜色/强度分布约束，避免生成图像出现不符合临床视觉特征的偏色。
- 用下游分类任务验证合成图像是否真正有助于医学 AI，而不只看生成图是否“像”。

## 局限性

这篇论文主要面向 2D RGB 医学图像，如胃肠镜和皮肤镜。对于乳腺 X 光、超声、MRI、病理切片等模态，仍需要重新考虑文本描述模板、图像归一化方式、临床真实性评价指标，以及是否应该从颜色约束扩展为灰度、纹理、结构或病灶区域约束。

此外，VLM 生成的描述可能存在医学事实不准确的问题，因此用于训练前最好加入专家校验、规则过滤或模态专用的描述生成策略。
