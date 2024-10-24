# OVTR: End-to-End Open-Vocabulary Multiple Object Tracking with Transformer
https://openreview.net/pdf?id=GDS5eN65QY

# Abstract
- 提出OVTR：
  - 设计CIP (Category Information Propagation) 策略，为后续帧建立multiple high-level类别信息先验
  - 设计了一个二分支结构，用于泛化能力和深度模态交互，并在decoder中加入保护策略以提高性能
- 优点：
  - 第一个端到端的开放词汇MOT模型
  - 值得注意的是，本方法不需要novel类别的proposals，但仍在开放词汇MOT基准上取得了很好的结果，此外，将模型转移到其他数据集的实验证明了其有效的适应性

# Method
## Overview
pipeline如图2所示
- 先构建了一个modality pre-fusion：输入从backbone获取的初步image features，以及从CLIP模型获取的text embeddings，执行pre-fusion生成fused image features I 和 text features T
- 在此之后，detect queries和track queries输入dual-branch decoder：该decoder受到attention isolation策略的保护，同时允许I和T的交互
- decoder的两个分支分别产生 $[O_{\mathrm{img}}^{\mathrm{det}},O_{\mathrm{img}}^{\mathrm{tr}}]$ 和 $[O_{\mathrm{txt}}^{\mathrm{det}},O_{\mathrm{txt}}^{\mathrm{tr}}]$ ：在这些输出中， $O_{txt}$ 参与和T的对比学习来进行分类， $O_{img}$ 作为类别信息传播 (CIP) 策略的输入，将当前帧的更新过的类别信息注入到类别信息流中
- 使用与 DETR 相同的方法将 $Q_{det}$ 与新出现目标匹配
<center><img src=../images/image-154.png style="zoom:70%"></center>

## Leveraging aligned queries for search in cross-attention
OVTR的感知部分建立于MOTR基础之上，在encoder和decoder中加入视觉语言模态融合。为高效进行多模态交互并学习泛化能力，decoder采用双分支结构，由OFA和CIT组成

### Generating Image and Text Embeddings
- 将text prompts送入到CLIP text encoder生成text embeddings
- 将ground-truths boxes送入CLIP image encoder生成image embeddings
  - 与使用来自RPN的部分包含novel类别的proposals生成大量image embeddings (如ViLD) 的方法不同，本方法的预处理更简单，并且不利用这些具有隐含novel类别的image embeddings信息
- CLIP生成的text和image embeddings是使用CLIP离线产生的

### Feature Pre-fusion and Enhancement
- 收到多模态检测器GLIP、Grounding DINO的启发，集成了image-to-text和text-to-image的cross-attention模块来进行特征融合，从而增强image和text的表征，为他们在decoder中的交互做好准备
- 由于enfcoder输出的初步content features可能会对decoder带来误导，因此遵循MOTR的做法，通过可学习的初始化生成queries的content part，而position part来自于encoder的输出

### Dual-Branch Structure
- 图3是双分支结构
  - 目标特征对齐 (object feature alignment, OFA) 分支包括：FFN后接一个box head和一个alignment head
  - 类别文本交互 (category text interaction, CTI) 分支包括：一个text cross-attention后接一个FFN
- 为了使模型实现zero-shot能力，利用OFA分支来对齐，引导image cross-attention层输出的queries，称其为aligned queries。<u>由于CLIP的image和text embeddings是对齐的，本方法将源自CLIP image embeddings的视觉泛化能力赋予到aligned queries中，使得它们能够有效地关注text cross-attention中text features传达的相应类别信息，甚至是引入novel类别时</u> (没整明白？似乎是泛化性的解释)具体的：
  - 从CLIP image encoder蒸馏知识来对齐alignment head的输出 $F_{align}$ 和CLIP image embeddings $V_{gt}$ ， $F_{align}$ 对应着二分匹配的结果，每个特征对应d维向量，n代表ground-truths目标的数量，alignment loss如下所示：
    <center><img src=../images/image-156.png style="zoom:70%"></center>

<center><img src=../images/image-155.png style="zoom:70%"></center>

<u>此外，双分支结构还旨在防止类别文本信息影响定位能力</u> (？突然来这么一句)

## Attention isolation for decoder protection
- 