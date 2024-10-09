# SLAck: Semantic, Location, and Appearance Aware Open-Vocabulary Tracking
https://arxiv.org/pdf/2409.11235  
# Abstract
- 以往的方法基于纯外观来匹配，由于开放词汇中运动模式的复杂性和novel目标分类的不稳定性，现有方法在最后匹配阶段，要么忽略运动和语义线索，要么根据启发式方法进行匹配
- 本文提出了一个联合的框架SLAck，在关联的早期阶段连接了语义、位置和外观先验，并通过一个轻量级的时间和空间目标图来学习如何联合所有有价值的信息
  - 本文方法无需使用复杂的后处理启发式方法来融合不同的线索，从而大大提高了大规模开放词汇跟踪的关联性能
  - 在MOT和TAO的TETA基准上达到了SOTA

# Introduction
- 现行的基于运动的MOT极度依赖于Kalman Filter，KF-based跟踪器依赖于线性运动假设，这在行人和车辆运动数据集中有效，但是由于变化的目标类别和运动模式，目标运动是非线性的，所以该假设在开放词汇场景中会失败
- 本文方法不同于显式KF-based运动，提出一个在关联学习过程中隐含位置先验的方法
  - 通过将位置和形状形式投影到特征空间并用多头注意力构建隐式的空间和时间目标图，本文的模型直接从数据中学习隐式的运动先验
  - 这种隐式建模使得模型能在空间上学习复杂的场景级目标结构，允许模型能在时间上捕捉线性和非线性运动
- 语义和运动之间的协同关系是明显的，因为运动模式通常和目标种类相关联
  - 如：一个模型在训练时学习到了马的运动模式，它可以在novel类别跟踪时，直接使用这样的知识迁移到和它语义类似的类别上，如斑马。在这种方法中，语义信息丰富了位置先验信息，即使不依赖外观线索，也能在novel类别跟踪中取得优异效果
- 外观模型在本文模型中通过加入融合embedding，被无缝集成到匹配过程中，如图2所示，这种方法与传统方法完全不同，传统方法通常在后期阶段采用基于启发式的融合方法
  <center><img src=../images/image-81.png style="zoom:50%"></center>

- 本文的联合模型，叫做SLAck，将语义、位置和外观封装在一个统一的关联框架中，用于开放词汇跟踪
  - 利用预训练的检测器，SLAck提取一整套描述符，并采用基于注意力的时空目标图 (STOG) 来促进信息交换，包括帧内的时间信息和跨帧的空间信息，类似的过程已被证明有助于为空间目标级关系建模，以进行目标检测 (论文Relation networks for object detection) 或计算点之间的对应关系 (论文Superglue)，该过程不仅提高了相关目标位置的感知能力，还使用语义信息对齐了运动和外观模式，从而获得了更加鲁棒的目标关联

# Method
## Preliminaries: Various cues for MOT
### Semantic cues
图3总结了过去MOT论文中利用语义信息的不同方法，语义线索在多类别MOT中经常起到的是比较小的作用，分类有：
- 通常使用硬分组，跟踪器根据检测器的预测将同一类别的目标关联起来 (如ByteTrack) ，这种方法在简单任务如跟踪人类或者车辆数据集 (KITTI、nuScenes)是高效的，问题：
  - 在开放词汇tracking中，分类是不可靠的 (如图2所示)，不确定的分类会影响tracking效果
- 软分组：TETer在特征空间中使用对比类别范例encodings做语义比较，从硬分组转为更可靠的软分组，问题：
  - 仍将语义信息归入基于启发式关联的后期阶段
- 本文方法：将语义线索尽早整合到关联过程中，利用其信息潜力提高学习和关联的准确性

<center><img src=../images/image-82.png style="zoom:50%"></center>

### Motion cues
代表性方法：SORT、OC-SORT、StrongSORT、ByteTrack，过去大多数基于运动的MOT方法基于线性运动假设的KF模型，线性运动模型的有效性是有限的，从DeepSORT+ViLD在Open-vocabulary上表现的效果就可以看出来，问题原因：
- 变化的相机视角
- 快速度的目标运动
- 跨类别的复杂的运动模式

本文方法：利用一种隐式运动建模的方法，即在目标之间建立时间和空间的关系，具体来说：
  - 将每个目标的位置和形状投影到特征空间，通过注意力机制实现帧内和帧间的交互，这一过程促进了目标之间关于其位置的信息交流，在不依赖明确线性假设的情况下增强了运动表示能力

### Appearance cues
代表性方法：TETer、MASA、OvTrack，这些方法在检测器上增加头并在静止图片或视频对上利用对比学习训练，检测器的头所输出的embedding用于关联，问题：
- 遮挡敏感
- 需要大量数据来学习鲁棒的匹配，这通常会导致在base类别上的过拟合
本文方法：在特征匹配过程的早期就将外观与语义信息整合在一起，利用语义的高层次语境，同时让外观头专注于较低层次的细节

### Hybrid cues
代表性方法：JDE、FairMOT、DeepSORT、ByteTrack，这些方法在最后匹配阶段，将一个空间距离矩阵和外观相似度矩阵使用启发式的方式融合在一起，用于匈牙利匹配  
本文方法：在早期就整合了所有有价值的信息，最终形成了一个奇异的匹配矩阵。这种早期融合避免了启发式的复杂性，提高了泛化能力，尤其是对novel类别的泛化能力

## Method Overview
- 基于预训练的开放或大型词汇检测器
- 从检测器中直接提取所有的信息，如语义、位置和外观，这些信息用一个时空目标图来推理出关联分配
- 本文方法是端到端的，无需任何额外的启发式方法来融合不同信息
- 借鉴Superglue论文，本文模型使用差分Sinkhorn-Knopp算法简单输出分配矩阵
- Detection Aware Training：直接使用检测框和TAO的稀疏Ground Truth作为关联学习的输入
- 完整的框架见图4

<center><img src=../images/image-83.png style="zoom:50%"></center>

## Extract Semantic, Location, and Appearance Cues
- 基于和TETer、OVTrack相同的目标检测器上构建跟踪器
- 在关联过程中冻结了所有检测器组件，以保持原有的强大开放词汇检测能力

### Semantic Head
- 使用和OVTrack相同的检测器，由于直接使用CLIP的encoder做语义线索会产生极高的推理成本，所以就对CLIP的文本encoder的知识蒸馏到RCNN分类头，用这个头来产生语义线索
- 在分类头后加一个5层MLP来将语义特征投影到最终的语义embedding $E_{sem}$ 
- 对于close-set的设定，使用TETer的检测器，使用其CEM编码作为语义头的输入

### Location Head
位置头将检测器的边界框头的输出作为输入，将其投影到特征空间中，具体的：
- 边界框坐标相对于图像尺寸进行归一化处理，以确保比例不变性，归一化过程包括相对于图像中心和尺寸的坐标缩放和平移
  - 给定边界框的坐标 $[x_{min}, y_{min}, x_{max}, y_{max}]$ ，图像尺寸 $[H,W]$
  - 图像中心 $(C_x,C_y)$ 的计算方法是 $C_x=\frac{W}{2}, C_y=\frac{H}{2}$ ，缩放比例为最大图像维度的 70%，即：$scaling = 0.7 \times max(H,W)$，归一化后的边界框的计算方法是：
    <center><img src=../images/image-94.png style="zoom:50%"></center>
  - 此归一化步骤确保图像内对象的空间位置相对于图像尺寸得以保留，从而促进尺度不变的位置特征
- 将归一化后的坐标输入到位置头来获得位置embedding $E_{loc}$
- 对于close-set的设定，将相应的置信度c和框坐标一起包含在内

### Appearance Head
外观头将RoI特征embeddings作为输入，输出针对关联优化过的外观embeddings，具体的：
- 外观头是一个简单的四层卷积，带有一个附加的MLP，输出为 $E_{app}$


## Spatial-Temporal Object Graph (STOG)
时空目标图STOG通过利用帧内self-attention和帧间cross-attention机制的组合 (follow的论文是Superglue)，编码了丰富的语义、位置和外观模式，这些模式对于理解开放世界tracking的目标动态是必要的

### Feature Fusion in STOG
- STOG前，每个目标通过一组不同但互补的特征进行表征： $E_{sem}$ $E_{loc}$ $E_{app}$
- STOG模型将每个目标的外观、位置和语义特征融合成统一的表征来初始化其过程，i表示帧内的第i个目标：
  <center><img src=../images/image-95.png style="zoom:50%"></center>
- 融合embeddings作为STOG的输入，通过帧内self-attention和帧间cross-attention机制进一步处理
  - 帧内的self-attention根据目标的相对位置和视觉相似性来完善对目标的理解
  - 帧间的cross-attention捕获目标的时间演化

### Intra-Frame Self-Attention
ljy：它定义了关键帧 (key frame) 和参考帧 (reference frame) ，应该一个是指轨迹的一帧，一个是指下一帧 (有检测)  
- 帧内self-attention独立处理关键帧和参考帧中的目标，来分析它们的空间关系和交互，对于keyframe (K) 和reference frame (R)，self-attention (SA) 的操作定义：
  <center><img src=../images/image-96.png style="zoom:50%"></center>
  
  其中， $\sigma$ 为softmax
- 此步骤通过关注每个目标与上下文最相关的特征，增强了模型对复杂的帧内目标排列和交互的理解

### Inter-Frame Cross-Attention
- 帧间cross-attention (CA)独立处理关键帧和参考帧，目的是跨帧对齐和更新目标特征，这捕捉了跟踪目标所必须的时间依赖性，目标从关键帧转换到参考帧以及反之的交叉注意操作被形式化为：
  <center><img src=../images/image-97.png style="zoom:50%"></center>

## Association Loss
给定一对帧，目标是计算这些帧中的目标之间进行正确匹配的损失。此过程涉及生成目标匹配矩阵，然后应用 Sinkhorn 算法来计算最优传输问题的可微近似。损失计算如下：

### Target Match Matrix Computation
对每对关键帧和参考帧构建一个二进制目标匹配矩阵T，表示跨帧目标之间的匹配，T的构建依赖于Ground Truth的匹配来确定目标之间的对应关系
- T中每个元素 $T_{ij}$ 为1或者0
- 为表示找不到匹配的目标 (目标消失或出现)，使用特殊的类‘dustbin’表示，将T扩展为T'，尺寸为 $(M+1) \times (N+1)$，M和N表示两个帧中的目标个数

### Training Loss
整个开放词汇跟踪框架可以使用 Sinkhorn 算法以端到端的方式进行训练，为最优传输问题提供了可微分的解决方案，具体的：
- 给定模型中的得分矩阵S (表示关键帧和参考帧中的对象之间的预测亲和力)，以及增强目标匹配矩阵 T'，Sinkhorn 损失 $L_{Sinkhorn}$ 计算如下：
  <center><img src=../images/image-98.png style="zoom:50%"></center>
  
  其中 $S'_{ij}$ 为应用了Sinkhorn迭代后的softmax归一化的分数矩阵，类似于概率匹配
  
## Detection Aware Training (DAT)
为了解决不完整的注释问题，同时又不影响我们预先训练的开放词汇检测器的检测能力，为我们在 TAO 视频训练期间冻结检测器的权重。这确保了探测器的性能保持不变。为了适应稀疏注释，我们采用了一种策略，检测器首先推断训练视频上的边界框，在训练和测试阶段保持输入数据的一致性。我们仅在这些预测框与可用的地面实况之间存在匹配时计算关联损失，忽略不匹配的对。事实证明，该方法对于 MOT 任务中的端到端训练非常有效，通过在训练过程中密切复制测试条件，可以显着提高性能

# Experiment
## 语义感知性能的消融实验
<center><img src=../images/image-99.png style="zoom:50%"></center>

## DAT性能的消融实验
<center><img src=../images/image-100.png style="zoom:50%"></center>

## 整合语义线索进行关联的不同方法比较
<center><img src=../images/image-101.png style="zoom:50%"></center>

## 开放词汇MOT对比试验
<center><img src=../images/image-102.png style="zoom:50%"></center>

## STOG性能的消融实验
<center><img src=../images/image-103.png style="zoom:50%"></center>

## Close-set MOT对比实验
<center><img src=../images/image-104.png style="zoom:50%"></center>