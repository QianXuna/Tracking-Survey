# Distilling DETR with Visual-Linguistic Knowledge for Open-Vocabulary Object Detection
https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Distilling_DETR_with_Visual-Linguistic_Knowledge_for_Open-Vocabulary_Object_Detection_ICCV_2023_paper.pdf
# Abstract
- 提出可以蒸馏视觉语言模型 (VLM，如CLIP) 的知识到DETR类似的检测器上，称为DK-DETR。具体的，提出两种巧妙的蒸馏方法：
  - 语义知识蒸馏（SKD）：显式地传递语义知识
  - 关系知识蒸馏（RKD）：利用目标之间的隐式关闭信息
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
      - 动机：虽然知识蒸馏可以有效提高新类别的性能，但它会对经过训练并具有足够的真实标签的base类别产生负面影响（例如表1中，ViLD 中的基本类别性能 $AP_f$ 从 32.5 下降到 30.3）。这种现象可归因于训练目标不一致以及VLM和检测器之间的域转移。在真实标签的监督下，检测器中的目标特征被训练来定位和识别base类别对象，但蒸馏迫使对象特征与 VLM 视觉嵌入保持一致，这会导致特征干扰
      - 本文添加了一组辅助查询来进行蒸馏，这避免了base类别的下降
        <center><img src=../images/image-76.png style="zoom:50%"></center>
    - 蒸馏方法和辅助查询仅仅用于训练，不会在推理中产生开销

# Method
DK-DETR的框架包括四个主要结构：组装基于 DETR 的文本嵌入检测器、辅助知识蒸馏分支、语义知识蒸馏和关系知识蒸馏。如图2所示
<center><img src=../images/image-77.png style="zoom:50%"></center>

## Overall Architecture
- 给一张图，encoder输出多尺度特征tokens作为memory特征
- 这些表示潜在目标的特征tokens被输入到分类头和回归头中，来生成目标置信度分数和粗略边界框
- 根据置信分数选择前N个tokens，并选相应的边界框 $B\mathrm{~}=\{\mathbf{b}_1,\mathbf{b}_2,\ldots,\mathbf{b}_N\}$ 作为初始anchor框
- 通过正弦encoder和投影层，这些锚框用于生成content queries $Q^{obj} = \{\mathbf{q}_1^{obj},\mathbf{q}_2^{obj},\ldots,\mathbf{q}_N^{obj}\} \in \mathbb{R}^{N\times D}$ （在DETR中被叫做object queries）和后边decoder的位置embedding
- 检测分支：
  - N个content queries、memory特征、位置embedding被输入到6个decoder层中，得到N个目标embeddings，得到N个潜在目标的特征
  - N个目标特征被送入了text-based分类器来产生对应于base类别分类分数（用于训练），或者base+novel类别的分类分数（用于推理），之后是一个用于将目标特征对齐到文本embedding维度的投影层，
  - 从object queries到分类分数的pipeline称为检测分支
- 此外，为了探索预训练视觉语言模型（VLM）中的丰富知识，引入了辅助知识蒸馏分支并提出了两种巧妙的知识蒸馏方案，即语义知识蒸馏和关系知识蒸馏知识蒸馏

## Text-based Classifier
Text-based分类器的步骤：
- 受预训练视觉语言模型CLIP的启发，我们建议通过类别名称本身来定义标注空间，如将一个名字做成句子后送入到VLM text encoder来提取文本embedding t
- 然后，embedding t去和目标的视觉特征f 计算余弦相似度：
    <center><img src=../images/image-78.png style="zoom:50%"></center>
- 最后，置信度分数为：
    <center><img src=../images/image-79.png style="zoom:50%"></center>

给定一个novel类别及其语言类别名称，Deformable DETR 可以轻松地与text-based分类器组装来执行 OVOD 任务，修改后的检测器作为本文的开放词汇目标检测baseline

## Auxiliary Distillation Branch
- 动机：当基线检测器直接推广到新类别时，由于缺少标注，检测性能不令人满意
- 本文提出蒸馏VLM image encoder的知识到检测器上
  - 给定从检测器encoder输出的粗略检测框B，我们在图像上截取区域并将其送入到VLM image encoder并提取特征 $V=V(crop(I,B))$ ，但是VLM和检测器的训练目标不一致（我理解是将检测器仅仅训练的是base+novel类，而VLM训练的则是所有类别），因此这种简单的知识蒸馏会引入干扰来干扰检测器特征