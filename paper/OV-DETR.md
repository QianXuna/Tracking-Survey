# Open-Vocabulary DETR with Conditional Matching
https://arxiv.org/pdf/2203.11876
# Abstract
- DETR转变为开放词汇检测的最大挑战：
  - 如果不获得novel类别的标注图像，就不可能算出novel类别的分类成本矩阵
  - 为克服该挑战，本文将学习目标设定为输入queries (类名称或示例图像) 和对应目标之间的二分匹配，它会学习有用的对应关系以泛化到测试期间未见过的queries
- 提出基于DETR的开放词汇检测器OV-DETR：
  - 可以检测给定类名或示例图像的任何目标
  - 训练时，扩展了DETR的decoder来接受输入条件queries，本文根据从预训练的视觉语言模型如CLIP获得的输入embedding来影响transformer的decoder，以便能够匹配文本和图像queries
# Introduction
- 现有工作：
  - 主要方法：如ViLD，中心思想是将检测器的特征与在CLIP模型提供的嵌入对齐（见图 1 (a)）。这样，就可以使用对齐的分类器仅从描述性文本中识别新类别
  - 存在问题：现有的依赖于RPN，生成的区域提议会不可靠地覆盖图像中地所有类别，容易过拟合，无法推广到新类别
- 本文方法：
  - 提出训练一个end-to-end地开放词汇检测器，目的是不使用中间的RPN来提高novel类别的泛化性
  - 困难：
<center><img src=../images/image-50.png style="zoom:50%"></center>

# Related Work
- Visual Grounding
  - 视觉Grounding任务是使用自然语言输入在一张图像中ground住一个目标
  - 视觉Grounding方法通常涉及特定的单个目标，因此不能直接应用于通用目标检测
  - 相关工作：MDETR。
    - 该方法类似地使用给定的语言模型来训练DETR，以便将DETR的输出标记与特定单词链接起来。 MDETR还采用了条件框架，将视觉和文本特征结合起来馈送到encoder和decoder
    - 然而，该方法不适用于开放词汇检测，因为它无法计算分类框架下新类的成本矩阵

# Open-vocabulary DETR
- 提出方法的动机：将具有Closed-set匹配的标准DETR改造为需要与未见过的类进行匹配的开放词汇检测器并非易事，这种开放集匹配的一种直观方法是学习一个类别无关的模块（例如ViLD）来处理所有类别
- 本文提供了DETR中匹配任务的新视角：将固定的集合匹配目标重新表述为condition输入（文本或图像查询）和检测输出之间的条件二进制匹配
- OV-DETR框架如图2所示：
  - DETR输入从CLIP模型获得的text或image的query embedding作为condition输入
  - 然后对检测结果施加二元匹配损失以衡量其匹配性
<center><img src=../images/image-56.png style="zoom:50%"></center>

## Revisiting Closed-Set Matching in DETR
DETR 管道的一次传递由两个主要步骤组成：
- 集合预测
- 最佳二分匹配

### Set Prediction
<center><img src=../images/image-57.png style="zoom:50%"></center>
<center><img src=../images/image-58.png style="zoom:50%"></center>

### Optimal Bipartite Matching
<center><img src=../images/image-59.png style="zoom:50%"></center>

### challenge
- 二分匹配方法不能直接应用到包含base和novel的开放词汇的设定中，原因是：计算3式需要获得标签信息，但是这对novel类别来说是不可获得的
- 如果参照先前的方法 (如ViLD) 生成类别无关的目标候选框，这些候选框可能会覆盖novel类别的目标，但问题仍然是不知道这些候选框的真实分类标签，所以结果仍然是目标queries的预测不能泛化到Novel的类别
- 如图3(a)所示，只能对具有可用训练标签的base类别执行二分匹配
<center><img src=../images/image-60.png style="zoom:50%"></center>

## Conditional Matching for Open-Vocabulary Detection
### Conditional Inputs
给定一个数据集，其中包含训练集的标注，需要将这些标注转换为条件输入，具体来说：
- 对每个有边界框 $b_i$ 和类别标签名 $y_i^{class}$ 的GT标注，使用CLIP生成对应的图像和文本embedding：
  <center><img src=../images/image-61.png style="zoom:50%"></center>
- 可以选择其中任何一个作为输入query来condition (意思可能是作为...的条件) DETR的decoder并训练以匹配相应的目标
- 训练完成后，可以在测试期间采用任意输入query来执行开放词汇检测
- 图像embedding $z_i^{image}$ 和文本embedding $z_i^{text}$ 作为query是等概率的，这是为了公平地学习
- 类似于先前的方法 (如ViLD) ，为novel类别目标生成额外的目标提案来丰富训练数据，对其仅提取图像embedding作为条件输入，因为训练集中它们的类别名是无法获得的

### Conditional Matching
本文核心训练目标是衡量条件输入embedding和检测结果之间的匹配度，具体来说：
- 用一个全连接网络将条件输入的embedding ( $z_i^{image}$ 或 $z_i^{text}$ )投影到和object query $q$ 有相同的维度，然后将2个embedding相加作为decoder的输入：
    <center><img src=../images/image-62.png style="zoom:50%"></center>

- 将条件输入embedding z仅添加到一个object query中将导致对可能在图像中多次出现的目标的覆盖范围非常有限，而实际的数据集有多个相同和不同类别的目标示例，为了丰富条件匹配的训练信号：
  - 本文在进行(5)式前，将object query q 复制了R次，将条件输入z复制了N次，最终获得了 $N\times R$ 大小的queries，如图4(b)所示，补充材料中的材料将证明这种特征复制的重要性以及如何确定N和R
    <center><img src=../images/image-63.png style="zoom:50%"></center>
    补充材料：
    <center><img src=../images/image-68.png style="zoom:50%"></center>
    <center><img src=../images/image-69.png style="zoom:50%"></center>
- 为了最后的condition过程，类似于UP-DETR，进一步添加了一个注意力mask来确保不同queries副本之间的独立性
- 标签分配的二进制匹配损失：
  <center><img src=../images/image-64.png style="zoom:50%"></center>

## Optimization
- 在式子(6)后，就获得了不同object queries的优化分配标签 $\sigma$
- 进一步将一个embedding reconstruction头加到模型中，该模型学习embedding e以便能够重建每个输入condition embedding
    <center><img src=../images/image-65.png style="zoom:50%"></center>

    补充材料证明了 $L_{embed}$ 的有效性
    补充材料：
    <center><img src=../images/image-72.png style="zoom:50%"></center>
- 模型的最终损失：
    <center><img src=../images/image-66.png style="zoom:50%"></center>

## Inference
<center><img src=../images/image-67.png style="zoom:50%"></center>

# Experiments
## Mask R-CNN和Deformable DETR的对比实验
<center><img src=../images/image-70.png style="zoom:50%"></center>

## 目标提案(P)和conditional binary matching机制(M)的消融实验
- 本文将Def DETR的分类层替换为了CLIP提供的文本embedding，并仅用base类别训练，这一步类似于ViLD-text的方法，如表2第1行所示
- 将可能包含novel类别的目标区域的提案加入到训练阶段，因为我们不知道这些目标提案的类别，所以我们观察到这些目标提案的标签分配不准确，如表2第2行所示
- 将DETR的默认闭集标签分配替换为我们提出的条件二元匹配，如表3第3行所示，表明二元匹配策略可以更好地利用目标提案中的知识

<center><img src=../images/image-71.png style="zoom:50%"></center>

## OV-LVIS、OV-COCO
<center><img src=../images/image-73.png style="zoom:50%"></center>

## Pascal VOC、COCO上的泛化性
<center><img src=../images/image-74.png style="zoom:50%"></center>