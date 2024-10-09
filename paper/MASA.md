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
    <center><img src=../images/image-105.png style="zoom:70%"></center>

- 创造自监督信号的方法：
  - 对同一图像应用不同的几何变换可以在同一图像的两个视角自动实现像素级对应
  - SAM的分割能力能够对同一实例的像素进行自动分组，从而实现像素级到实例级对应关系的转换
  - 该过程利用视图对之间的密集相似性学习，创建用于学习有区别的目标表示的自监督信号。本文方法的训练策略允许使用来自不同领域的丰富的原始图像集合
- 在自监督训练pipeline之外，进一步构建了一个通用的tracking adapter：MASA适配器，以支持任何现有的开放世界分割和检测foundation model，如SAM、Detic和Grounding-DINO来跟踪它们检测到的任何目标。为保留其分割和检测能力，冻结了原始的backbone，在顶部添加了MASA adapter
- 提出了一个多任务训练pipeline，联合执行SAM的检测知识的蒸馏和示例相似性学习，<u>这种方法能够先于SAM学习目标的位置、形状和外观，并在对比相似性学习期间模拟真实的检测候选框</u> (没懂)

# Related Work
## Learning Instance-level Association
学习实例级对应关系，现有的方法分为：
- 自监督：UniTrack
- 监督：SHUSHI、TETer、Trackformer、QDTrack、OVTrack、GTR等等有很，OVTrack、GTR等也需要在静态图像中学习跟踪信号，仍然需要特定领域中大量标注

# Method
## Preliminaries: SAM
- SAM由三个模块组成：
  - image encoder：用于特征提取的ViT-based backbone
  - prompt encoder：对来自点、框或mask的prompts进行建模
  - mask decoder：transformer-based decoder 输入提取的图像embedding与级联输出和prompt tokens，进行最终的mask预测

## Matching Anything by Segmenting Anything
训练pipeline如图2所示
<center><img src=../images/image-106.png style="zoom:70%"></center>

### MASA Pipeline
- 核心思想：从两个角度增加多样性：训练图像多样性和实例多样性
  - 如图1所示，首先构建来自不同领域的丰富原始图像集合，以防止学习特定于领域的特征。这些图像还包含复杂环境中的丰富实例，以增强实例多样性
  - 给定图像，采用两种不同的增强来模拟视频中的外观变化，构建两个不同的视图，自动获得像素级对应关系
  - 如果图像是干净的并且仅包含一个实例，例如 ImageNet 中的实例，则可以应用帧级相似性，然而，对于多个实例，需要进一步挖掘这些原始图像中包含的实例信息。SAM 自动对属于相同实例的像素进行分组，并提供检测到的实例的形状和边界信息
- SAM 对整个图像的详尽分割会自动生成密集且多样化的实例建议 Q 集合。建立像素级对应关系，对Q应用 $\phi(\cdot)$ 和 $\psi(\cdot)$ ，得到自监督信号，借鉴论文Supervised Contrastive Learning，使用对比学习公式来学习判别性对比学习embedding空间
    <center><img src=../images/image-107.png style="zoom:50%"></center>
    负样本对于学习判别性表示至关重要。在对比学习范式下，SAM 生成的密集候选框自然地提供更多的负样本，从而增强学习更好的关联实例表示

### MASA Adapter
- 目的：扩展开放世界分割和检测模型 (如SAM、Detic、Grounding-DINO) 来跟踪任何检测到的目标
- 方法：
  - MASA 适配器与这些基础模型的冻结主干特征结合使用，确保保留其原始检测和分割功能，然而，由于并非所有预先训练的特征本质上都具有跟踪区分性，因此首先将这些冻结的主干特征转换为更适合跟踪的新特征
  - 考虑到目标形状和大小的多样性，构建了一个多尺度特征金字塔
    - 对于像Detic、Grounding-DINO中的Swin Transformer的分层backbones，直接使用FPN
    - 对于使用普通ViT backbone的SAM，使用transpose convolution (反卷积) 和 maxpooling 对步长为16的单尺度特征进行上采样和下采样，来生成尺度比为1/4, 1/8, 1/16, 1/32 的分层特征
  - 为了有效地学习不同实例的区分特征，一个位置的目标必须了解其他位置的实例的外观，像DynamicHead一样，使用Deformable Convolution来生成动态offset并聚合跨空间位置和特征层的信息：
      <center><img src=../images/image-108.png style="zoom:50%"></center>
      
      其中L表示特征层数，K为卷积核的采样位置个数，wk和pk分别为卷积核的权重和预定义的第k个位置的offset， $\Delta p_k^j$ 和 $\Delta m_k^j$ 分别为第k个位置第j层特征层的可学洗的offset和调制因子
  - 获取变换后的特征图后，通过将 RoI-Align 应用到视觉特征 F 来提取实例级特征，然后使用包含 4 个卷积层和 1 个全连接层的轻量级轨道头进行处理以生成实例embedding
  - 对于基于SAM的模型：
    - <u>还使用DynamicHead的任务感知注意力和尺度感知注意力，因为检测性能对于准确的自动mask生成非常重要，如图3(b)所示</u> 没理解
    - 在训练期间引入了一个目标先验蒸馏分支作为辅助任务。该分支采用标准 RCNN 检测头来学习紧密包含 SAM 对每个实例的mask预测的边界框。它有效地从 SAM 中学习详尽的对目标位置和形状知识，并将这些信息蒸馏为转换后的特征表示。这种设计不仅增强了MASA适配器的功能，从而提高了关联性能，而且还通过直接提供预测的框提示来加速SAM的一切模式
  - Loss：使用和Faster RCNN中定义的检测损失和之前定义的对比损失来优化： $L=L_{det}+L_C$ 

### 推理
图3为test pipeline  
#### Detect and Track Anything
- 将MASA adapter和目标检测器集成时，移除MASA在训练时学习到的检测头，MASA adapter仅充当跟踪器，输出track features
- 检测器预测边界框，利用边界框prompt MASA adapter，adapter检索相应的跟踪特征以进行实例匹配，使用bi-softmax最近邻搜索来实现实例匹配:
    <center><img src=../images/image-110.png style="zoom:50%"></center>
    <center><img src=../images/image-111.png style="zoom:50%"></center>
    <center><img src=../images/image-112.png style="zoom:50%"></center>
<center><img src=../images/image-109.png style="zoom:70%"></center>

#### Segment and Track Anything
- 将MASA adapter和SAM集成时，保留检测头，MASA adapter不仅输出track features，还输出检测框
- 将预测框作为prompt传递给SAM的mask decoder和MASA adapter，分别来分割任意目标和跟踪任意目标 (我的理解：MASA adapter的检测框再输入到adapter用于输出track features)，预测框的prompt省略了原始SAM的everything mode中繁重的后处理的需要，所以显著加快了SAM的自动mask生成的速度

#### Test with Given Observations
当从MASA adapter所基于的以外的来源获取检测结果时，MASA adapter将充当跟踪功能提供者。直接利用边界框作为提示，通过 ROI-Align 操作从MASA adapter中提取跟踪特征

# Experiments
## TAO TETA基准上的对比试验
<center><img src=../images/image-113.png style="zoom:50%"></center>

## TAO 开放词汇MOT上的对比试验
<center><img src=../images/image-114.png style="zoom:50%"></center>

## TAO Track mAP基准上的对比试验
<center><img src=../images/image-115.png style="zoom:50%"></center>

## BDD MOTS基准上的对比试验
<center><img src=../images/image-116.png style="zoom:50%"></center>

## BDD MOT基准上的对比试验
<center><img src=../images/image-117.png style="zoom:50%"></center>