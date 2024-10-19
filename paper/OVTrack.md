# OVTrack: Open-Vocabulary Multiple Object Tracking
https://arxiv.org/pdf/2304.08408
# Abstract
- 提出OVTrack
  - 通过知识蒸馏利用视觉语言模型进行分类和关联
  - 从去噪扩散概率模型中学习鲁棒外观特征的数据幻觉策略
- 数据效率极高的跟踪器
- 在TAO基准上达到了SOTA
- 仅在静态图像上训练

# Introduction
- Open-world Tracking (单目标) 过去使用recall-based的评估，问题：
  - 无法惩罚误报 (FP)
  - 类别无关的方式评估tracking，不能评估一个跟踪器能否正确推理一个目标的语义类别
- 本文定义了Open-vocabulary MOT，其旨在跟踪超越预定义训练类别之外的多个目标，但本文没有忽略分类问题，也没有利用基于召回的评估，而是假设在测试时给定感兴趣的目标分类，这允许能够应用现有的closed-set tracking指标来获取精确率和召回率，同时仍然能够评估跟踪器在推理过程中跟踪任意目标的能力
- 本文确定并解决了开放词汇多目标跟踪器设计的两个基本挑战
  - closed-set MOT方法无法扩展其预定义的分类
  - 数据可用性，即将视频数据标注扩展到大量类词汇量的成本极高
- 本文提出了第一个开放词汇跟踪器 OVTrack，如图1所示
  - 检测相关：受到开放词汇检测现有工作的启发，用嵌入头替换了分类器，使得能够衡量本地目标与语义类别的开放词汇表的相似性。具体地，通过将目标提案的图像特征表征与相应的CLIP图像和文本嵌入对齐，将CLIP中的知识蒸馏到我们的模型中
  - 关联相关：
    - 运动：由于Open-vocabulary包含任意的场景、任意的相机模式、任意的目标运动模式，运动特征通常是脆弱的
    - 外观：外观特征在多样性的目标上会呈现截然不同的特征，而依赖外观线索需要能够推广到novel目标类别的模型，而本文发现CLIP特征蒸馏能学到更好的外观表征来提升MOT的关联
- 学习鲁棒的特征需要强大的监督来捕获目标在不同的视角、背景和光亮的变化，为解决该问题，本文提出使用DDPM生成一张静态图片的带有随机背景扰动的正例和负例样本
<center><img src=../images/image-40.png style="zoom:50%"></center>

# Related Work
- Multiple object tracking
  - 过去的方法像GTR、AOA、QDTrack、TETer受限于预先设定的目标种类，所以不能扩展到真实世界多样性的设定中，而本文的工作允许跟踪开放词汇表中未见过的种类
- Open-world detection and tracking
  - 开放世界检测
    - 利用类不可知的定位器并将分类视为聚类问题，估计新实例之间的相似性并通过增量学习将它们分组为新
  - 开放词汇检测
    - 旨在在测试时检测任意的，但给定的目标类别
    - 很多工作利用CLIP的图文表征能力，用于开放词汇和few-shot目标检测
      - ViLD蒸馏CLIP图像特征
      - Detic利用CLIP分类数据进行联合训练
      - DetPro重点是学习开放词汇目标检测的良好语言prompts
    - 开放世界跟踪
      - TAO论文存在的局限性是评估仅捕捉跟踪器的召回率，而没有分类准确率，但本文没有忽略分类问题，在测试阶段给定感兴趣的novel类别
- Learning tracking from static images
  - 相关工作：
    - CenterTrack：提出通过输入的随机平移来学习静态图像的运动偏移
    - FairMOT：将静态图像数据集中的目标视为唯一的类来区分
    - QDTrack：建议利用数据增强与对比学习目标相结合，从静态图像中学习基于外观的跟踪
  - 本文通过生成模型生成对象的正面和负面示例以及背景扰动，提供了更有针对性的方法来指导从静态图像中学习外观相似性
- Data generation for tracking
  - DDPM利用其数据生成保真度来解决数据可用性问题，该问题在开放词汇MOT中尤为明显，并采用针对外观建模量身定制的新颖数据幻觉策略

# Open-Vocabulary MOT
## Task设定
<center><img src=../images/image-41.png style="zoom:50%"></center>

## Benchmark
- Dataset：TAO
- 分类法：TAO主要遵循LVIS的分类法，LVIS根据出现情况将类别分为frequent, common, rare classes。
  - Base class: frequent class + common class
  - Novel class：rare class
- Metric：
  - 过去使用的Track mAP不能评估FP
  - TETA可以除了能评估定位和关联，还可以处理分类

# OVTrack
## Model design
模型设计架构如图3所示
<center><img src=../images/image-45.png style="zoom:50%"></center>

### Localization
类别无关方法训练Faster R-CNN

### Classification
- 现存的closed-set跟踪器仅跟踪base类别的目标，即训练数据分布中出现的目标类别
- 为了实现开放词汇分类，需要能够配置我们感兴趣的类而无需重新训练。受到开放词汇检测文献ZSD的启发，将Faster R-CNN和CLIP联系起来
  - 提取RoI的特征 $f_r = \mathcal{R}(\phi(I),\mathbf{b}_r),\forall r\in P$ ，r为RPN proposal输出得目标候选集合P的一个元素
  - 将Faster R-CNN的分类头换为一个文本头，还加了一个图像头，用于为每个$f_r$生成embeddings $\hat{t_r}$ 和 $\hat{i_r}$ ，用CLIP的文本和图像的encoder来监督这2个头 (两个头都是从CLIP知识蒸馏的)。具体地：
    - 对于文本头：
      - 使用类别名生成包括L个文本向量 $v_c$ 和一个类名embedding $w_c$ 的文本prompt集合 $\mathcal{P}(c) = \{\mathbf{v}_1^c,...,\mathbf{v}_L^c,\mathbf{w}_c\}$ 
      - 将prompt集合输入到CLIP的文本encoder中， $\mathbf{t}_c=\mathcal{E}(\mathcal{P}(c)),\forall c\in\mathcal{C}^\text{base}$ 
      - 计算 $\hat{t_r}$ 和 $t_c$ 之间的相似性
          <center><img src=../images/image-42.png style="zoom:50%"></center>
      - $t_{bg}$ 为一个可学习的背景prompt，$\lambda$ 为温度参数， $L_{CE}$ 为交叉熵损失， $c_r$ 为r的类别标注
    - 对于图像头：
      - 用CLIP的图像encoder，对类别r，crop出一个输入图像为 $b_r$ ，获得图像embedding $\mathbf{i}_r=\mathcal{I}(\mathcal{R}(I,\mathbf{b}_r))$ ，同理最小化 $\hat{i_r}$ 和 $i_r$ 之间的距离
        <center><img src=../images/image-43.png style="zoom:50%"></center>

### Association
- 由于Open-vocabulary包含任意的场景、任意的相机模式、任意的目标运动模式，运动特征通常是脆弱的
- 本文依靠外观线索来稳健地跟踪开放词汇上下文中的目标
  - 受到QDTrack、TETer的启发，本文使用了一种对比学习方法：
    - 给定一对图片对 $(I_{key}, I_{ref})$ ，从两个图像中提取RoIs，并使用IoU将RoIs与标注进行匹配
    - 对 $I_{key}$ 中每个匹配到的具有外观embedding $q\in Q$ 的RoI，将具有相同ID的放到一个集合 $Q^+$ 中，不同ID的放到一个集合 $Q^-$ 中：
        <center><img src=../images/image-44.png style="zoom:50%"></center>
        公式来源参考TETer和 Supervised Constractive Learning
    - 我们进一步应用QDTrack提到的辅助损失 $L_{aux}$ 来约束下面的logits的大小
### Infer pipelne
<center><img src=../images/image-47.png style="zoom:50%"></center>
双向softmax见论文QDTrack，bi-softmax下的高分表明两个匹配目标是特征空间中彼此最近的邻居
<center><img src=../images/image-46.png style="zoom:50%"></center>

## Learning to track without video data
DDPM

# Experiments
## Open-vocabulary MOT task: TAO val set and test set
<center><img src=../images/image-48.png style="zoom:50%"></center>

- Closed-set跟踪器：在 $C^{base}\cup C^{novel}$ 的类别上训练的
- Open-vocabulary跟踪器：在 $C^{novel}$ 的类别上训练的
- OVTrack：只见过静态图数据 （LVIS）
- 最后一栏将RegionCLIP与baseline方法和OVTrack方法结合起来，其中用RegionCLIP替换OVTrack的定位和分类的部分

## Closed-set MOT task: TAO val
<center><img src=../images/image-49.png style="zoom:50%"></center>

问题：
- 该问题是针对所有开放词汇跟踪器的问题
  - 怎么训练RPN？RPN的目的是获得所有不是背景的类别无关的候选框，即它的分类是二值的：背景/前景，那么就有问题了：训练集不能有Base类别的，其具体形式有以下可能：
    1. 训练图片只有base类别的目标，且有base类别的标注。
    2. 训练图片有base和novel类别的目标，且只有base类别的标注
    如果是第一种情况，RPN学习的“背景”只有背景元素；如果是第二种情况，RPN学习的“背景”包括背景和novel类别元素，这就有问题了，其在推理时将novel类别判定为背景怎么办？
    为了确定是哪一种，最应该做的是去看ViLD的任务设置，其任务setting是相似的
 - 回答：参考ViLD的问题部分
