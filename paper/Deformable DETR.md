# Deformable DETR: Deformable Transformers for End-to-End Object Detection
https://arxiv.org/pdf/2010.04159

# Abstract
- 提出Deformable DETR
  - 结合可变形卷积的稀疏空间采样的优点和Transformers的关系建模能力
  - 提出可变形注意力模块，关注一小组采样位置作为所有特征图像素中关键元素的预过滤器
  - 该模块可以自然扩展到聚合多尺度特转增，而无需FPN
- 在小目标上比DETR有更好地性能，缓解DETR收敛速度慢和复杂度高的问题，训练次数减少10倍
- COCO

# Introduction
Deformable DETR如图1所示：
<center><img src=../images/image-131.png style="zoom:50%"></center>

# REVISITING TRANSFORMERS AND DETR
## Multi-Head Attention in Transformers
多头注意力结构：
<center><img src=../images/image-132.png style="zoom:50%"></center>
<center><img src=../images/image-133.png style="zoom:50%"></center>

Transformer的问题：
<center><img src=../images/image-134.png style="zoom:50%"></center>

