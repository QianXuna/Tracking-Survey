# Deformable DETR: Deformable Transformers for End-to-End Object Detection
https://arxiv.org/pdf/2010.04159

# ABSTRACT
- 提出Deformable DETR
  - 结合可变形卷积的稀疏空间采样的优点和Transformers的关系建模能力
  - 提出可变形注意力模块，关注一小组采样位置作为所有特征图像素中关键元素的预过滤器
  - 该模块可以自然扩展到聚合多尺度特转增，而无需FPN
- 在小目标上比DETR有更好地性能，缓解DETR收敛速度慢和复杂度高的问题，训练次数减少10倍
- COCO

# INTRODUCTION
Deformable DETR如图1所示：
<center><img src=../images/image-131.png style="zoom:50%"></center>

# REVISITING TRANSFORMERS AND DETR
## Multi-Head Attention in Transformers
多头注意力结构：
<center><img src=../images/image-132.png style="zoom:50%"></center>
<center><img src=../images/image-133.png style="zoom:50%"></center>

Transformer的问题：
<center><img src=../images/image-134.png style="zoom:50%"></center>

## DETR
DETR结构：
<center><img src=../images/image-135.png style="zoom:50%"></center>

DETR的问题：
<center><img src=../images/image-136.png style="zoom:50%"></center>

<center><img src=../images/image-137.png style="zoom:50%"></center>

# METHOD
## DEFORMABLE TRANSFORMERS FOR END-TO-END OBJECT DETECTION
### Deformable Attention Module
可变形注意力模块仅关注参考点周围的一小组key采样点，而不管特征图的空间大小，结构如图2所示，公式如下所示：
<center><img src=../images/image-139.png style="zoom:50%"></center>
复杂度：
<center><img src=../images/image-140.png style="zoom:50%"></center>
<center><img src=../images/image-141.png style="zoom:50%"></center>
<center><img src=../images/image-138.png style="zoom:70%"></center>

### Multi-scale Deformable Attention Module
大多数现代目标检测框架受益于多尺度特征图，本文提出的可变形注意力模块能够自然地扩展到多尺度特征图，公式如下所示：
<center><img src=../images/image-142.png style="zoom:50%"></center>

- 当L=1，K=1且 $W_m'$ 固定为单位矩阵时，可变形注意力模块退化为可变形卷积模块，可变形卷积为单尺度特征图输入设计的，每个注意力头仅关注一个采样点
- 本文的多尺度可变形注意力会关注来自多尺度输入的多个采样点
- 所提出的（多尺度）可变形注意模块也可以被视为 Transformer 注意力有效变体，其中通过可变形采样位置引入预过滤机制。当采样点遍历所有可能的位置时，所提出的注意力模块相当于Transformer注意力

### Deformable Transformer Encoder
用多尺度可变形注意力模块取代DETR的Transformer注意力模块，具体而言：
- 多尺度特征图的构建：
  - 从ResNet的 $C_3$ 到 $C_5$ 阶段的输出特征图中提取多尺度特征图 $\{x^l\}_{l=1}^{L-1} (L=4)$ (通过1*1卷积)，其中 $C_l$ 的分辨率比输入图像第 $2^l$ 
  - 最低分辨率特征图 $x^L$ 通过 $C_5$ 阶段的 3*3 stride=2 的卷积获得，表示为 $C_6$
  - 所有多尺度特征图有C=256个通道
  - 没有使用FPN中的自上而下结构，因为根据式(3)可以看到，多尺度可变形注意力本身可以在多尺度特征图之间交换信息
- encoder的结构：
  - 输入和输出都是有相同分辨率的多尺度特征图，key元素和query元素都是多尺度特征图中的元素，对于每个query像素，参考点是它本身
  - 为了识别每个query像素所处的特征层，除了位置特征向量外，还在特征表征中添加了一个尺度级特征向量 $e_l$ ，不同于固定编码的位置向量，尺度级向量 $\{e_l\}_{l=1}^L$ 为随机初始化的，且与网络共同训练
- 结构如附录A.2：
    <center><img src=../images/image-143.png style="zoom:50%"></center>

### Deformable Transformer Decoder
- decoder中包括cross-attention和self-Attention
  - 将cross-attention替换为多尺度可变形注意力，self-attention不变
  - 参考点 $\hat{p_q}$ 的2d标准化的坐标是由object query embedding通过一个可学习的线性投影层+sigmoid函数预测到的
- 由于多尺度可变形注意力模块提取的是参考点周围的图像特征，因此让家安侧头预测边界框与参考点的相对偏移量，以进一步降低优化难度
  - 参考点被用作边界框中心的初始猜测
  - 检测头预测参考点的偏移，详见附录A.3
  - 作用：学习到的decoder注意力将与预测的边界框有很强的相关性
  <center><img src=../images/image-144.png style="zoom:50%"></center>

## ADDITIONAL IMPROVEMENTS AND VARIANTS FOR DEFORMABLE DETR