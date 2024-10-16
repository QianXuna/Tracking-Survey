# Distilling DETR with Visual-Linguistic Knowledge for Open-Vocabulary Object Detection
https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Distilling_DETR_with_Visual-Linguistic_Knowledge_for_Open-Vocabulary_Object_Detection_ICCV_2023_paper.pdf
# Abstract
- 提出可以蒸馏视觉语言模型 (VLM，如CLIP) 的知识到DETR类似的检测器上，称为DK-DETR。具体的，提出两种巧妙的蒸馏方法：
  - 语义知识蒸馏（SKD）：显式地传递语义知识
  - 关系知识蒸馏（RKD）：利用目标之间的隐式关系信息
  - 此外，检测器中添加了包含一组辅助查询的蒸馏分支，以减轻对base类别的负面影响
- 加入SKD和RKD，提高了novel类别的检测性能，避免了对base类别的检测
- LVIS、COCO达到SOTA

# Introduction
- knowledge distillation (KD)：
  - 将新类别目标的知识从VLM提炼到检测器是解决开放词汇目标检测（OVOD）问题的典型做法，如ViLD，如图1(a)所示
  - ZSD-YOLO、HierKD用知识蒸馏技术在一阶段检测器上提升novel类别的表现，而不是ViLD上用的二阶段
  - 端到端检测器也出现，但将知识蒸馏到端到端检测器在OVOD领域还没怎么研究
    <center><img src=../images/image-75.png style="zoom:50%"></center>

- 提出新框架
  - 先前方法：使用的经典知识蒸馏方法导致在novel类别上的提升有限
  - 本文方法：
    - 提出了SKD和RKD，如图1所示
      - 在SKD中，检测器和VLM image encoder之间的特征对齐被看成pseudo-classification(伪分类)问题，而不是普通知识蒸馏中的回归问题，不仅将属于同一目标的特征聚集在一起，而且将来自不同目标的特征分开
      - 在RKD中，考虑到VLM能在丰富的视觉实体之间构建结构良好的特征空间，本文提出将VLM图像编码器中隐藏的目标之间的关系进行建模，并将关系知识蒸馏到检测器中
    - 提出辅助查询
      - 动机：虽然知识蒸馏可以有效提高新类别的性能，但它会对经过训练并具有足够的真实标签的base类别产生负面影响（例如表1中，ViLD 中的基本类别性能 $AP_f$ 从 32.5 下降到 30.3）。这种现象可归因于训练目标不一致以及VLM和检测器之间的域转移。在真实标签的监督下，检测器中的目标特征被训练来定位和识别base类别对象，但蒸馏迫使目标特征与 VLM 视觉嵌入保持一致，这会导致特征干扰
      - 本文添加了一组辅助查询来进行蒸馏，这避免了base类别的下降
        <center><img src=../images/image-76.png style="zoom:50%"></center>
    - 蒸馏方法和辅助查询仅仅用于训练，不会在推理中产生开销

# Method
DK-DETR的框架包括四个主要结构：组装基于 DETR 的文本嵌入检测器、辅助知识蒸馏分支、语义知识蒸馏和关系知识蒸馏。如图2所示
<center><img src=../images/image-77.png style="zoom:50%"></center>

## Overall Architecture
整体流程如下：
- 给一张图，encoder输出多尺度feature tokens作为memory features
- 表示潜在目标的feature tokens被输入到分类头和回归头中，来生成目标置信度分数和粗略边界框
  - **Deformable的two_stage模式下，在first stage，即encoder时，会生成archers (类似于RPN生成archers)，像RPN一样，针对archers做分类和回归，得到Proposals，分类的目的只是区分前景与背景，所以生成的proposals是类别无关的，是具有除了背景以外的所有base和novel类别的粗略的框，可以参考OVTrack在 4.1 Model Design的Localization部分提到的：它使用了RPN和回归损失，这种定位过程可以很好地推广到训练时novel的目标类别上**
    > We find that this localization procedure can generalize well to object classes that are unknown at training time, as also validated by previous works [11, 22, 79].
    --OVTrack, page 4
- 根据置信分数选择前N个tokens，并选相应的边界框 $B\mathrm{~}=\{\mathbf{b}_1,\mathbf{b}_2,\ldots,\mathbf{b}_N\}$ 作为初始的anchor框
- 通过正弦encoding和投影层，这些archor框用于生成content queries $Q^{obj} = \{\mathbf{q}_1^{obj},\mathbf{q}_2^{obj},\ldots,\mathbf{q}_N^{obj}\} \in \mathbb{R}^{N\times D}$ （在DETR中被叫做object queries）以及为后边decoder生成positional embedding
- 在图2中，从content queries到classification scores的pipeline称为检测分支
  - N个content queries (相当于DETR的object queries)、memory features (相当于DETR encoder输出的embeddings)、positional embeddings被输入到6个decoder层中，得到N个object embeddings，即N个潜在的object features
  - 使用投影层将N个object features和text embedding的尺寸对齐，接着对齐后的object features被送入了text-based分类器来产生对应于base类别classification scores（用于训练），或者base+novel类别的classification scores（用于推理）
- 此外，为了探索预训练视觉语言模型（VLM）中的丰富知识，引入了辅助知识蒸馏分支并提出了两种巧妙的知识蒸馏方案，即语义知识蒸馏SKD和关系知识蒸馏RKD
- **框的定位：根据阅读DK-DETR的源代码，其配置文件中明确说明了所使用的decoder为六层DetrTransformerDecoderLayer，每层Decoder都包括：self-attention、cross-attention、FFN，每层Decoder都会预测相对于参考点的唯一，基于DeformableDETR的Iterative Bounding Box Refinement机制，在decoder的最后一层就会输出一个迭代优化过的准确的框的定位**

## Text-based Classifier
text-based分类器的步骤：
- 受预训练视觉语言模型CLIP的启发，本文通过"类别名称"来定义label space (标注空间)，如将一个名字做成句子后送入到VLM text encoder来提取text embedding **t**
- 再用embedding **t**去和object feautres **f**计算余弦相似度：
    <center><img src=../images/image-78.png style="zoom:50%"></center>
- 最后，置信度分数为：
    <center><img src=../images/image-79.png style="zoom:50%"></center>

给定一个novel类别及其语言上的类别名称，Deformable DETR 可以轻松地与text-based分类器组装来执行 OVOD 任务，修改后的检测器作为本文的开放词汇目标检测baseline

## Auxiliary Distillation Branch (辅助蒸馏分支)
- 动机：当baseline检测器直接推广到novel类别时，由于缺少标注，检测性能不令人满意
- 本文提出蒸馏VLM image encoder的知识到检测器上
  - 给定从检测器encoder输出的粗略检测框B，在图像上截取对应区域并将其送入到VLM image encoder并提取特征 $V=V(crop(I,B))$ ，但是VLM和检测器的训练目标不一致（我理解是将检测器仅仅训练的是base+novel类，而VLM训练的则是所有类别），因此这种简单的知识蒸馏会引入干扰来扰乱检测器特征
  - 蒸馏分支：
    - 本文在蒸馏分支中引入了一组辅助可学习的embeddings，叫做KD queries $Q^{kd} = \{\mathbf{q}_1^{kd},\mathbf{q}_2^{kd},\ldots,\mathbf{q}_N^{kd}\} \in \mathbb{R}^{N\times D}$ ，如图2中所示，它对应着 $Q^{obj}$ ，它们共享相同的positional embeddings
    - **蒸馏分支和检测分支共享相同的网络权重**，输入KD queries和positional embeddings，decoder生成KD features $F^{kd}$ ，用于下文中的蒸馏
    - 注意：decoder的self-attention模块从KD queries到object queries的attention masks被阻止，避免影响到 object features

## Semantic Knowledge Distillation
- 目的：将VLM iamge encoder $V(\cdot)$ 的知识直接蒸馏到检测器上
- 方法：
  - 过去方法：像ViLD，一一对齐特征
    - Loss：计算两种features之间的直接损失 (即L1 Loss) 来对齐features，以L1 loss为监督的公式：
      <center><img src=../images/image-118.png style="zoom:50%"></center>
    - 问题：
      - 仅以一对一方式对齐特征，没有利用VLM中足够信息
      - 严格的对齐增加了训练难度

  - 本文方法：
    - 将普通知识蒸馏中的回归问题重新表述为伪分类问题，如图3所示，属于同一目标的一组KD feature和VLM视觉embeddings为正例对，否则为负例对
      - Loss：用余弦相似度计算分类分数，二元交叉熵 (BCE) loss 将正例样本punish为标签1，负例样本punish为标签0，以 BCE loss为监督的公式：
        <center><img src=../images/image-120.png style="zoom:50%"></center>
      - 优点 (相比L1 loss)：
        - 不会punish蒸馏分支的output features和VLM image encoder的visual embeddings完全相同，降低训练难度
        - L1 loss仅将相同目标的两种features拉近，而SKD loss不仅如此，还能将不同目标的featuress推的很远，能利用一些隐式的关系信息

<center><img src=../images/image-119.png style="zoom:50%"></center>

## Relational Knowledge Distillation
- 动机：
  - SKD是对两种features的直接对齐
  - 两个目标之间的relation反映它们的correspondence，如在VLM embedding space中，一个老虎可能比狗更接近猫，如果将这些知识蒸馏给检测器，它可能帮助检测器在推理过程中避免将老虎识别为狗
- 目的：建模并蒸馏图片中不同个体目标之间的关系
- 本文方法：如图4所示，给定VLM的visual embeddings **V** 和 KD features $F^{kd}$，用两个成对相似矩阵来分别表示VLM features和KD features的关系： VLM关系图 $R^v=\bar{F}\bar{F}^T\in\mathbb{R}^{N\times N}$，KD关系图 $R^{kd}=\bar{V}\bar{V}^T\in\mathbb{R}^{N\times N}$，其中 $\bar{F}$ 和 $\bar{V}$ 分别为L2标准化过的 **V** 和 $F^{kd}$ ，这两个相似性矩阵捕获了目标之间的成对相关性，我们引导来自蒸馏分支的矩阵 $R^{kd}$ 与来自 VLM 图像编码器的 $R^v$ 对齐：
  <center><img src=../images/image-122.png style="zoom:50%"></center>

  - 值得注意的是，单个图像中的目标之间的关系只是可以捕获的一种相关性。还有其他关系也可以帮助模型，例如来自不同图像的目标之间的关系，这将在消融实验中进一步讨论
  <center><img src=../images/image-121.png style="zoom:50%"></center>

## Loss Functions
<center><img src=../images/image-123.png style="zoom:50%"></center>
<center><img src=../images/image-124.png style="zoom:50%"></center>

# Experiments
## LVIS数据集对比试验
<center><img src=../images/image-126.png style="zoom:50%"></center>

## COCO开放词汇数据集对比试验
<center><img src=../images/image-127.png style="zoom:50%"></center>

## 泛化能力实验
<center><img src=../images/image-128.png style="zoom:50%"></center>

## SKD、RKD、KD queries性能的消融实验
<center><img src=../images/image-129.png style="zoom:50%"></center>

## RKD不同关系类型的消融实验
<center><img src=../images/image-125.png style="zoom:50%"></center>

问题：
- 代码实现，整体流程怎么做
- 局部问题：
  - 用于蒸馏decoder的content queries为什么要去掉base类别，只留下novel类别？:
    <center><img src=../images/image-130.png style="zoom:50%"></center>