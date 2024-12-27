# Simple Image-Level Classification Improves Open-Vocabulary Object Detection

## Abstract

开放词汇对象检测 (OVOD) 旨在检测超出训练检测模型的给定基本类别集的新对象。最近的 OVOD 方法侧重于通过区域级知识蒸馏、区域提示学习或区域级等方法，将图像级预训练视觉语言模型 (VLM)（例如 CLIP）适应区域级目标检测任务。文本预训练，扩大检测词汇量。这些方法在识别区域视觉概念方面表现出了显着的性能，但在利用 VLM 从数十亿级图像级文本描述中学到的强大的全局场景理解能力方面却很薄弱。这限制了它们从新颖/基本类别中检测小、模糊或遮挡外观的硬物体的能力，其检测严重依赖于上下文信息。为了解决这个问题，我们提出了一种新颖的方法，即**用于上下文感知检测评分的简单图像级分类（Simple Image-level Classification for Context-Aware Detection Scoring, SIC-CADS）**，以利用 CLIP 产生的卓越全局知识从全局角度补充当前的 OVOD 模型。 SIC-CADS的核心是**多模态（multi-modal）多标签识别（multi-label recognition, MLR）模块**，该模块从CLIP中学习基于 **目标共现 (co-occurrence-based)** 的上下文信息，以识别场景中所有可能的目标类别。然后，可以利用这些图像级 MLR 分数来细化当前 OVOD 模型在检测这些硬物体时的实例级检测分数。这已通过两个流行基准 OV-LVIS 和 OV-COCO 的大量实证结果得到验证，这些结果表明 SIC-CADS 在与不同类型的 OVOD 模型结合时取得了显着且一致的改进。此外，SIC-CADS还提高了Objects365和OpenImages上的跨数据集泛化能力。代码可在 <https://github.com/mala-lab/SIC-CADS> 获取。

## Introduction

- 相关研究：
  - 现有的基于VLM的OVOD研究重点是如何使图像级预训练的CLIP适应到区域级目标检测任务。现有的方法有通常采用区域概念学习方法，如：
    - 区域级知识蒸馏DetPro
    - 区域提示学习CORA、DetPro
    - 区域文本预训练RegionCLIP
    - 自训练Detic
- 对方法的攻击：
  - 这些方法在识别区域视觉概念方面表现出了显着的性能，但在利用 VLM 强大的全局场景理解能力来捕获不同视觉概念之间的重要关系方面却很薄弱。这些关系可以是：**co-occurrence 共生关系**：如网球和网球拍、**inter-dependence 相互依赖关系**：如网球和网球场。这一弱点可能会限制它们检测外观较小、模糊或被遮挡的**hard objects**的能力，这些hard objects的检测很大程度上**依赖于同一图像中其他物体的上下文特征**。
  - 当将区域特征与相应的文本嵌入对齐时，OV 检测器自然地学习了**base类别**的上下文信息，因为网络可以自动捕获与base目标相关的感受野内的上下文特征。然而，由于在 OV 检测器的训练过程中**缺乏novel目标标注**，因此无法学习与novel目标相关的上下文特征。这可能是造成base目标和novel目标（尤其是novel的hard objects）之间性能差距的关键原因之一
- 本文解决问题的方法
  - 动机：为了解决这个问题，这项工作旨在利用 OVOD 的全局场景理解能力，动机是**从 CLIP 的图像编码器中提取的图像级嵌入携带了整个场景中各种目标的全局特征，这些特征在自然语言描述中在语义上相关。然后，这些知识可以为检测上述hard objects（例如图 1（c）中的小而模糊的网球）提供重要的上下文信息**，否则仅使用区域特征很难检测到这些hard objects
  - 方法：受此启发，我们提出了一种利用简单图像级分类模块进行上下文感知检测评分的新颖方法，称为 SIC-CADS。我们的图像级分类任务由多标签识别（MLR）模块指定，该模块学习从 CLIP 中提取的多模态知识。 MLR 模块预测特定场景中可能存在的不同可能目标类别的图像级分数。例如，如图1（d）所示，网球拍和网球场的上下文信息有助于识别此类运动相关场景中的模糊网球。因此，图像级MLR分数可用于细化现有OVOD模型的实例级检测分数，以从全局角度提高其检测性能，如图1（e）所示。
- 本文贡献：我们的主要贡献总结如下。 (i) 我们提出了一种新颖的方法 SIC-CADS，该方法利用 MLR 模块来利用 VLM 的全局场景知识来提高 OVOD 性能。 (ii) SIC-CADS 是一个简单、轻量级的通用框架，可以轻松插入不同的现有 OVOD 模型，以增强其检测硬目标的能力。 (iii) 对 OV-LVIS、OV-COCO 和跨数据集泛化基准的大量实验表明，SIC-CADS 与不同类型的最先进 (SOTA) OVOD 模型结合使用时，可显着提高检测性能，实现OV-LVIS 的 APr 增益为 1.4 - 3.9，OV-COCO 的 APnovel 增益为 1.7 - 3.2。此外，我们的方法还大大提高了它们的跨数据集泛化能力，在 Objects365 上获得了 1.9 - 2.1 的 mAP50 增益（Shao et al. 2019），在 OpenImages 上获得了 1.5 - 3.9 的 mAP50 增益（Kuznetsova et al. 2020）。

<center><img src=../images/image-166.png style="zoom:70%"></center>

## Related Work

- 零样本多标签识别（Zero-Shot Multi-Label Recognition, ZS-MLR）：多标签识别的任务是预测一张图像中所有目标的标签。作为零样本学习 (Zero-Shot Learning, ZSL) 的扩展（Romera-Paredes 和 Torr 2015；Xian、Schiele 和 Akata 2017；Xian 等人 2019；Xu 等人 2020），ZS-MLR 旨在识别 **可见（seen）** 和 **不可见（unseen）** 的目标图像中的目标。 ZS-MLR 的关键是将图像嵌入与类别嵌入对齐。之前的 ZS-MLR 方法（Zhang、Gong 和 Shah 2016；Ben-Cohen 等人 2021；Huynh 和 Elhamifar 2020；Narayan 等人 2021；Gupta 等人 2021）大多使用语言模型中的单模态嵌入（例如，GloVe（Pennington、Socher 和 Manning 2014））并采用不同的策略（例如，注意力机制（Narayan et al. 2021；Huynh and Elhamifar 2020）或生成模型（Gupta et al. 2021））来提高性能。He等人（He et al. 2022）进一步提出了开放词汇多标签识别（OV-MLR）任务，该任务采用来自 CLIP 的多模态嵌入来实现 MLR。他们使用 Transformer 主干提取区域和全局图像表征，并使用双流（two-stream）模块从 CLIP 传输知识。识别任何模型（Recognize Anything Model, RAM）（Zhang et al. 2023）通过解析自动图像标记的标题来利用大规模图像文本对。然后，他们训练了一个图像标签识别解码器，该解码器显示出强大的 ZS-MLR 性能。

- 分辨open-vocabulary和zero-shot：
  <center><img src=../images/image-168.png style="zoom:70%"></center>

> 来源：<https://www.zhihu.com/question/68275921/answer/2034667107>
综述：<https://arxiv.org/pdf/2307.09220>
github 知识库：<https://github.com/seanzhuh/awesome-open-vocabulary-detection-and-segmentation>

## Method

这项工作旨在利用 CLIP 的图像级/全局知识来更好地识别硬物体，从而实现更有效的 OVOD。我们首先通过从 CLIP 传输全​​局多模态知识来训练多标签识别（MLR）模块，以识别整个场景中的所有现有类别。然后在推理过程中，我们的 MLR 模块可以通过简单的检测分数细化过程轻松插入现有的 OVOD 模型，以提高检测性能。

### Preliminaries

<center><img src=../images/image-170.png style="zoom:70%"></center>
### CLIP-driven Multi-modal MLR Modeling

### CLIP-driven Multi-modal MLR Modeling

#### Overall Framework

- 输入图片，使用 ResNet50-FPN backbone 提取多尺度特征图，然后使用全局最大池化 (Global Max Pooling, GMP) 运算符和不同 FPN 级别中的所有特征向量的串联以获得一个全局图像嵌入 $e^{global}$
- 随后在两个分支中使用全局图像嵌入：包括一个 text MLR 分支，用于对齐 $e^{global}$ 和 CLIP text encoder产生的不同类别的嵌入，和一个 visual MLR 分支，用于把 CLIP image encoder产生的全局图像嵌入蒸馏给 $e^{global}$ 。在推理过程中，text MLR 在识别 novel 类别方面较弱，因为它仅使用 base 类别进行训练，而visual MLR 通过蒸馏 CLIP 的 zero-shot 识别 (recognition) 能力可以识别 base 类别和 novel 类别。因此，我们结合两个分支的 scores 来实现 novel 类别和 base 类别的更好的识别性能，这被称为多模态 MLR。下面我们详细介绍一下它们。

<center><img src=../images/image-167.png style="zoom:70%"></center>

#### Text MLR

<center><img src=../images/image-169.png style="zoom:70%"></center>

#### Visual MLR

<center><img src=../images/image-171.png style="zoom:70%"></center>

<center><img src=../images/image-172.png style="zoom:70%"></center>

#### Multi-modal MLR

<center><img src=../images/image-173.png style="zoom:70%"></center>
<center><img src=../images/image-174.png style="zoom:70%"></center>

### Context-aware OVOD with Image-Level Multi-modal MLR Scores

<center><img src=../images/image-175.png style="zoom:70%"></center>
