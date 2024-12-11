# OVTR: End-to-End Open-Vocabulary Multiple Object Tracking with Transformer
https://openreview.net/pdf?id=GDS5eN65QY

# Introduction
- OVTrack的问题：
  - 每帧中独立进行目标分类，这导致类别感知的潜在不稳定，并阻止在后续帧中重复使用先前预测的结果
  - 相似的外观、外观变化和不可预测的运动模式往往会破坏掉严重依赖外观特征的现有策略的有效性
  - 这种TBD框架不可避免地依赖于后处理和archor生成，这需要基于特定场景先验知识的手工设计操作，使得它们难以适应开放世界环境
- 提出OVTR：
  - 设计CIP (Category Information Propagation) 策略，为后续帧建立multiple high-level类别信息先验
  - 设计了一个二分支结构，用于泛化能力和深度模态交互，包括OFA分支和CTI分支
    - OFA分支：目标特征对齐分支。利用CLIP来对齐OFA分支输出的表征并生成文本表征来作为CTI分支的输入；OFA分支引导模型获得新类别的视觉泛化能力，确保queries的对齐并生成object-level的类别表示作为 CIP 策略的输入
    - CTI分支：类别文本交互分支。CTI分支通过cross-attention促进这些对齐的queries和类别文本特征之间的交互
  - 在decoder中加入保护策略以提高性能，目的是减少queries之间的内容差异造成的干扰，使得分类和跟踪能够协同工作，引入了两种调节self-attention的策略：
    - 类别隔离策略：作用是保护decoder免受因类别信息差异引起的潜在混淆
    - 内容隔离策略：作用是保护decoder免受因跟踪和检测queries之间的相互作用而引起的潜在混淆
- 优点：
  - 第一个端到端的开放词汇MOT模型
  - 本方法不需要novel类别的proposals，但仍在开放词汇MOT基准上取得了很好的结果，此外，将模型转移到其他数据集的实验证明了其有效的适应性

# Method
## Overview
pipeline如图2所示
### Revisiting MOTR
这部分重新回顾的MOTR的逻辑结构
- 第一帧 $f_{t=1}$ ，detect queries $Q_{det}^{t=1}$ 被输入到decoder中，在这里和encoder输出的图像特征 $E_{img}^{t=1}$ 交互，该过程生成更新后的detect queries $Q_{det}^{'t=1}$ ，这些queries包含目标信息
- 然后从 $Q_{det}^{'t=1}$ 中提取出检测的预测结果，包括了边界框 $B_{det}^{t=1}$ 和目标表征 $O_{det}^{t=1}$
- 和DETR相反的是，在该跟踪器上， $Q_{det}^{t=1}$ 仅仅需要检测当前帧新出现的目标，于是通过进行二元匹配的方式，仅仅在 $Q_{det}^{'t=1}$ 和新出现目标的ground truth之间做一对一的分配
- 匹配到的 $Q_{det}^{'t=1}$ 将被用于更新和生成 track queries $Q_{tr}^{t=2}$ ，这些queries用于第二帧 $f_{t=2}$ 并被再次送入decoder和图像特征 $E_{img}^{t=2}$ 交互来提取和 $Q_{tr}^{t=2}$ 匹配的目标的位置和表征，从而进行跟踪预测
- 随后， $Q_{tr}^{t=2}$ 被更新进而为第三帧 $f_{t=3}$ 生成 $Q_{tr}^{t=3}$ ,像 $f_{t=1}$ 帧一样，并行于 $Q_{tr}^{t=2}$ 的 $Q_{det}^{t=2}$ 输入到decoder来检测新出现的目标，在二分匹配后， $Q_{det}^{'t=2}$ 被转换为新的track queries，并添加到 $Q_{tr}^{t=3}$ 中
- 以上整个的跟踪过程能够扩展到后续的帧中
- 关于优化问题，MOTR应用了多帧优化，loss的计算考虑了ground truths和matching results，每帧的matching results包括维护的track associations、 $Q_{'det}$ 和新出现目标之间的二分匹配

### Tracking Mechanism During Inference
与 MOTR 类似，OVTR 推理期间的网络前向过程遵循与训练期间相同的过程。主要区别在于轨道查询的转换。在检测预测中，如果类别置信度得分超过τdet，则相应更新的检测查询将转换为新的轨迹查询，从而启动新的轨迹。相反，如果跟踪对象在当前帧中丢失（置信度≤τtr），则将其标记为非活动跟踪。如果非活动轨道丢失了 Tmiss 连续帧，则它会被完全删除。
### Empowering Open Vocabulary Tracking
利用基于queries的框架的迭代特性，OVTR 跨帧传输有关跟踪目标的信息，在整个连续图像序列中聚合类别信息以实现强大的分类性能，而不是在每帧中执行独立的定位和分类。
- 在encoder中，来自backbone的初始图像特征和来自CLIP模型的文本embeddings通过pre-fusion来生成融合图像特征 $E_{img}$ 和融合文本特征 $E_{txt}$
- dual-branch decoder包括OFA分支和CTI分支，当输入 $Q=[Q_{det},Q_{tr}]$ 时，两个分支分别引导 Q 导出 (derive) 视觉泛化表征 (visual generalization representations) 、与 $E_{txt}$ 进行深度模态交互 (deep modality interaction) ，分别输出 $O_{img}$ 和 $O_{txt}$ 
- $O_{img}$ 作为类别信息传递 (Category information propagation, CIP) 机制的输入，将类别信息注入到类别信息流中，该过程是对MOTR中机制的扩展，即从 $Q_{det}^{'t}$ 生成 $Q_{tr}^{t+1}$ 
- $O_{txt}$ 用于计算类别逻辑 (category logits) 和对比学习

<center><img src=../images/image-157.png style="zoom:70%"></center>

## Leveraging aligned queries for search in cross-attention
OVTR的感知部分建立于MOTR基础之上，在encoder和decoder中加入视觉语言模态融合。为高效进行多模态交互并学习泛化能力，decoder采用双分支结构，由OFA和CIT组成。

### Generating Image and Text Embeddings
- 将text和prompts送入到CLIP text encoder生成text embeddings
- 将ground-truths boxes送入CLIP image encoder生成image embeddings，并将同一类别的embeddings组合成单个表征
  - 不同于使用 (额外的RPN检测器生成的) 包含了novel类别目标的proposals来生成大量image embeddings的方法 (如ViLD、DetPro) ，我们的预处理更简单，我们也不利用具有隐式novel类别信息的图像embeddings，CLIP生成的text和image embeddings都是离线产生的

### Feature Pre-fusion and Enhancement
- 受到多模态检测器GLIP、Grounding DINO的启发（这在GLIP中称为Language-Aware Deep Fusion，即使用跨模态多头注意力机制X-MHA融合图像特征和文本特征），我们在encoder中集成了image-to-text和text-to-image的cross-attention模块来进行特征融合，从而增强image和text的表征，为他们在decoder中的交互做好准备
- 由于encoder输出的初步content features可能会对decoder带来误导，因此我们遵循 ~~MOTR~~ DINO-DETR 的做法，即通过可学习的初始化 (learnable initialization) 生成queries的content part，通过encoder的输出 $E_{img}$ (利用sin-cos positional encoding)产生的reference points 生成queries的position part

### Dual-Branch Structure
- 图3是双分支结构
  - 目标特征对齐 (object feature alignment, OFA) 分支包括：FFN后跟着一个box head和一个alignment head
  - 类别文本交互 (category text interaction, CTI) 分支包括：一个text cross-attention后接一个FFN
- 为了使模型实现zero-shot能力，利用OFA分支来对齐，引导image cross-attention层输出的queries，称其为**aligned queries**。 **由于CLIP的image和text embeddings是对齐的，本方法将源自CLIP image embeddings的视觉泛化能力赋予到aligned queries中，使得它们能够有效地关注text cross-attention中text features传达的相应类别信息。这是因为 $E_{txt}$ 源自于CLIP的text embeddings。直观上，即使引入novel类别的目标，与文本特征 $E_{txt}$ 具有相同类别信息的aligned queries也会隐式地与它们对齐。** 具体而言：
  - 从CLIP image encoder蒸馏知识,将alignment head的输出 $F_{align} \in R^{n \times d}$ 和CLIP image embeddings $V_{gt}\in R^{n \times d}$ 对齐， $F_{align}$ 对应着'Overview'节中提到的二分匹配的结果，每个特征对应d维向量，n代表ground-truths目标的数量，alignment loss $L_{align}$如下所示：
    <center><img src=../images/image-156.png style="zoom:70%"></center>

<center><img src=../images/image-158.png style="zoom:70%"></center>
<u>此外，双分支结构还旨在防止类别文本信息影响定位能力</u> (？突然来这么一句)【我的理解：定位能力在OFA分支上，文本信息在CTI分支上，设计两个分支使得定位使用的表征不会杂糅】

## Attention isolation for decoder protection
- 对于decoder，在多个类别信息和track queries的内容都可能产生干扰。具体来说：
  - self-attention中的queries之间的交互可能会纠缠类别信息，从而对分类性能产生负面影响
  - decoder并行地处理track queries和detect queries。track queries包含有关被跟踪目标的内容，从而在它们和初始化的detect queries之间出现内容gap。由于self-attention的相互作用，这种gap可能会导致decoder层内之间的冲突
- 为了解决以上问题，我们提出了注意力隔离机制以用于decoder protection

### Category Isolation Strategy
### Content Isolation Strategy

