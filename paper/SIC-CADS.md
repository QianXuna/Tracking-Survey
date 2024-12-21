# Simple Image-Level Classification Improves Open-Vocabulary Object Detection
## Abstract
开放词汇对象检测 (OVOD) 旨在检测超出训练检测模型的给定基本类别集的新对象。最近的 OVOD 方法侧重于通过区域级知识蒸馏、区域提示学习或区域级等方法，将图像级预训练视觉语言模型 (VLM)（例如 CLIP）适应区域级目标检测任务。文本预训练，扩大检测词汇量。这些方法在识别区域视觉概念方面表现出了显着的性能，但在利用 VLM 从数十亿级图像级文本描述中学到的强大的全局场景理解能力方面却很薄弱。这限制了它们从新颖/基本类别中检测小、模糊或遮挡外观的硬物体的能力，其检测严重依赖于上下文信息。为了解决这个问题，我们提出了一种新颖的方法，即**用于上下文感知检测评分的简单图像级分类（Simple Image-level Classification for Context-Aware Detection Scoring, SIC-CADS）**，以利用 CLIP 产生的卓越全局知识从全局角度补充当前的 OVOD 模型。 SIC-CADS的核心是**多模态多标签识别（multi-modal multi-label recognition, MLR）模块**，该模块从CLIP中学习基于对象共现的上下文信息，以识别场景中所有可能的对象类别。然后，可以利用这些图像级 MLR 分数来细化当前 OVOD 模型在检测这些硬物体时的实例级检测分数。这已通过两个流行基准 OV-LVIS 和 OV-COCO 的大量实证结果得到验证，这些结果表明 SIC-CADS 在与不同类型的 OVOD 模型结合时取得了显着且一致的改进。此外，SIC-CADS还提高了Objects365和OpenImages上的跨数据集泛化能力。代码可在 https://github.com/mala-lab/SIC-CADS 获取。

## Introduction
- 相关研究：
  - 现有的基于VLM的OVOD研究重点是如何使图像级预训练的CLIP适应到区域级目标检测任务。现有的方法有通常采用区域概念学习方法，如：
    - 区域级知识蒸馏DetPro
    - 区域提示学习CORA、DetPro
    - 区域文本预训练RegionCLIP
    - 自训练Detic
- 对方法的攻击：
  - 这些方法在识别区域视觉概念方面表现出了显着的性能，但在利用 VLM 强大的全局场景理解能力来捕获不同视觉概念之间的重要关系方面却很薄弱。这些关系可以是：**co-occurrence 共生关系**：如网球和网球拍、**inter-dependence 相互依赖关系**：如网球和网球场。这一弱点可能会限制它们检测外观较小、模糊或被遮挡的**hard objects**的能力，这些hard objects的检测很大程度上依赖于同一图像中其他物体的上下文特征
  - 当将区域特征与相应的文本嵌入对齐时，OV 检测器自然地学习了**base类别**的上下文信息，因为网络可以自动捕获与base目标相关的感受野内的上下文特征。然而，由于在 OV 检测器的训练过程中**缺乏novel目标标注**，因此无法学习与novel目标相关的上下文特征。这可能是造成base目标和novel目标（尤其是novel的hard objects）之间性能差距的关键原因之一
- 本文解决问题的方案
  - 动机：为了解决这个问题，这项工作旨在利用 OVOD 的全局场景理解能力，动机是从 CLIP 的图像编码器中提取的图像级嵌入携带了整个场景中各种目标的全局特征，这些特征在自然语言描述中在语义上相关。然后，这些知识可以为检测上述hard objects（例如图 1（c）中的小而模糊的网球）提供重要的上下文信息，否则仅使用区域特征很难检测到这些hard objects
  - 