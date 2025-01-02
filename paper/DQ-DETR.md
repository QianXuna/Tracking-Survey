# Dual Query Detection Transformer for Phrase Extraction and Grounding

https://arxiv.org/pdf/2211.15516

## Introduction

- Visual Grounding包括下属方向Referring Expression Comprehension（REC）、Phrase Grounding、Phrase Extraction and Grounding（PRG）
  - REC：使用自由格式的引导文本来定位目标，它只根据指代表达式（referring expression）的要求检测一类目标，如图1 (b)所示。
  - Pharse Grounding：需要找到表达式中提到的所有目标，如图1 (c)所示。因为短语（pharse）在测试过程中被假定为已知，Pharse Grounding可以通过提取短语（extract pharse）作为指代表达式来重新表述为REC任务。
  - PEG：本文认为，在测试过程中把短语视为未知则更为实际，如图1 (d)所示
    - 将现有 REC 模型扩展到 PEG 的一个简单方法是开发一个两阶段解决方案：首先使用 spaCy 等 NLP 工具提取短语，然后应用REC模型。然而，这样的解决方案可能会导致性能较差，因为两个阶段之间没有交互。例如，图像可能不具有与提取的短语相对应的目标或包含多个目标。然而，大多数 REC 模型对于每个提取的短语仅预测一个目标。更不用说不准确的短语提取可能会误导 REC 模型来预测不相关的目标。
    - 我们注意到，短语提取是从输入文本中定位名词短语，这可以被视为预测目标短语的一维文本掩码的一维文本分割问题。这样的问题类似于 2D 图像分割中目标实例的 2D 掩模预测。受到类 DETR 模型（例如 DINO、Mask2Former）最近进展的启发，本文开发了一种更有原则的解决方案 DQDETR，这是一个基于用于PEG任务的双重查询的类DETR模型。如图 2 (d) 所示，我们的模型使用双重查询在一个 DETR 框架中执行目标检测和文本掩码预测。文本掩码预测与 Mask2Former 中的实例掩码预测非常相似，因此我们可以使用 masked-attention Transformer 解码器层来提高文本掩码预测的性能。在 DQ-DETR 中，一对双重查询被设计为具有共享位置部分（position part）但不同内容部分（content part）。这种解耦的查询设计有助于减轻图像和文本之间模态对齐的难度，从而产生更快的收敛和更好的性能。

<center><img src=../images/image-176.png style="zoom:70%"></center>

<center><img src=../images/image-177.png style="zoom:70%"></center>