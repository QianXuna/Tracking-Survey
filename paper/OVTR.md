# OVTR: End-to-End Open-Vocabulary Multiple Object Tracking with Transformer
https://openreview.net/pdf?id=GDS5eN65QY

# Abstract
- 提出OVTR：
  - 设计CIP (Category Information Propagation) 策略，为后续帧建立multiple high-level类别信息先验
  - 设计了一个二分支结构，用于泛化能力和深度模态交互，包括OFA分支和CTI分支
    - OFA分支：目标特征对齐分支，OFA分支引导模型获得新类别的视觉泛化能力，确保queries对齐并生成object-level的类别表示作为 CIP 策略的输入
    - CTI分支：类别文本交互分支，CTI分支通过cross-attention促进这些对齐的queries和类别文本特征之间的交互
  - 在decoder中加入保护策略以提高性能，目的是减少queries之间的内容差异造成的干扰，使得分类和跟踪能够协同工作，有两种策略：
    - 类别隔离策略
    - 内容隔离策略
- 优点：
  - 第一个端到端的开放词汇MOT模型
  - 本方法不需要novel类别的proposals，但仍在开放词汇MOT基准上取得了很好的结果，此外，将模型转移到其他数据集的实验证明了其有效的适应性

# Introduction
- OVTrack的问题：
  - 每帧中独立进行目标分类，这导致类别感知的潜在不稳定，并阻止在后续帧中重复使用先前预测的结果
  - 相似的外观、外观变化和不可预测的运动模式往往会破坏掉严重依赖外观特征的现有策略的有效性
  - 这种TBD框架不可避免地依赖于后处理和archor生成，这需要基于特定场景先验知识的手工设计操作，使得它们难以适应开放世界环境
  

# Method
## Overview
pipeline如图2所示
- 先构建了一个modality pre-fusion，便于后续的交互，具体来说：encoder输入从backbone获取的初步image features，以及从CLIP模型获取的text embeddings，执行pre-fusion生成fused image features I 和 text features T
- 在此之后，detect queries和track queries并行输入到dual-branch decoder：该decoder受到attention isolation策略的保护，同时允许I和T的交互
- decoder的两个分支分别产生 $[O_{\mathrm{img}}^{\mathrm{det}},O_{\mathrm{img}}^{\mathrm{tr}}]$ 和 $[O_{\mathrm{txt}}^{\mathrm{det}},O_{\mathrm{txt}}^{\mathrm{tr}}]$ ：在这些输出中， $O_{txt}$ 参与和T的对比学习来进行分类， $O_{img}$ 作为类别信息传播 (CIP) 策略的输入，将当前帧的更新过的类别信息注入到类别信息流中
- 使用与 DETR 相同的方法将detect queries $Q_{det}$ 与新出现目标匹配，用 $O_{img}$ 来更新detect queries，得到track queries $Q_{tr}$ ，用于表征它们的轨迹。推理期间，当 $Q_{tr}$ 没有被匹配时，就被删掉，表明预测轨迹的消失
<center><img src=../images/image-154.png style="zoom:70%"></center>

## Leveraging aligned queries for search in cross-attention
OVTR的感知部分建立于MOTR基础之上，在encoder和decoder中加入视觉语言模态融合。为高效进行多模态交互并学习泛化能力，decoder采用双分支结构，由OFA和CIT组成

### Generating Image and Text Embeddings
- 将text prompts送入到CLIP text encoder生成text embeddings
- 将ground-truths boxes送入CLIP image encoder生成image embeddings
  - 与使用来自RPN的部分包含novel类别的proposals生成大量image embeddings (如ViLD) 的方法不同，本方法的预处理更简单，并且不利用这些具有隐含novel类别的image embeddings信息
- CLIP生成的text embeddings和image embeddings是使用CLIP离线产生的

### Feature Pre-fusion and Enhancement
- 受到多模态检测器GLIP、Grounding DINO的启发（这在GLIP中称为Language-Aware Deep Fusion，即使用跨模态多头注意力机制X-MHA融合图像特征和文本特征），集成了image-to-text和text-to-image的cross-attention模块来进行特征融合，从而增强image和text的表征，为他们在decoder中的交互做好准备
- 由于encoder输出的初步content features可能会对decoder带来误导，因此遵循MOTR的做法，通过可学习的初始化生成queries的content part，而position part来自于encoder的输出

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