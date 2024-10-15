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
- 

### Deformable Transformer Decoder

## ADDITIONAL IMPROVEMENTS AND VARIANTS FOR DEFORMABLE DETR