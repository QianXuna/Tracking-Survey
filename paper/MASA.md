# Matching Anything by Segmenting Anything
https://arxiv.org/pdf/2406.04221
# Abstract
- 提出MASA，一个用于鲁棒实例关联学习的新颖方法，能够跨不同域匹配视频中的任何目标，无需tracking标注
  - 利用SAM的丰富目标分割，MASA通过详尽的数据增强来学习实例级对应关系
    - 将SAM的输出视为密集的目标候选区域，并学习从大量图像集合中去匹配这些区域
  - 设计了一个通用的MASA适配器，它可以与分割foundation model或检测模型协同工作，使它们能跟踪任何检测到的目标
- MASA在复杂领域中有强大的zero-shot跟踪能力，所提出的方法仅使用未标注的静态图像，在zero-shot关联中比使用全部标注的域内视频序列训练的最先进方法实现了更好的性能

# Introduction
- 目标是：
  - 开发一种能够匹配任何目标或区域的方法
  - 将这种通用的跟踪功能与任何检测和分割方法集成，以实现跟踪检测到的任何目标
- 主要挑战：获得不同领域的一半目标的匹配监督，且不产生大量的标注成本
- 提出了 Matching Anything by Segmenting Anything (MASA) 的pipeline，来从任何域的未标注图像上学习object-level关联，如图1所示
    <center><img src=../images/image-105.png style="zoom:50%"></center>

- 创造自监督信号的方法：
  - 对同一图像应用不同的几何变换可以在同一图像的两个视角自动实现像素级对应
  - SAM的分割能力能够对同一实例的像素进行自动分组，从而实现像素级到实例级对应关系的转换
  - 该过程利用视图对之间的密集相似性学习，创建用于学习有区别的目标表示的自监督信号。本文方法的训练策略允许使用来自不同领域的丰富的原始图像集合
- 在自监督训练pipeline之外，进一步构建了一个通用的tracking adapter：MASA适配器，以支持任何现有的开放世界分割和检测foundation model，如SAM、Detic和Grounding-DINO来跟踪它们检测到的任何目标。为保留其分割和检测能力，冻结了原始的backbone，在顶部添加了MASA adapter
- 提出了一个多任务训练pipeline，                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            