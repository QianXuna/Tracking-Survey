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
- 确定并解决了开放词汇多目标跟踪器设计的两个基本挑战
  - closed-set MOT 方法无法扩展其预定义的分类方法
  - 数据可用性，即将视频数据注释扩展到大量类词汇量的成本极高
- 提出了第一个开放词汇跟踪器 OVTrack，如图1所示
  - 受到开放词汇检测现有工作的启发，用嵌入头替换了分类器，使得能够测量本地化对象与语义类别的开放词汇表的相似性。特别是，通过将目标提案的图像特征表示与相应的CLIP图像和文本嵌入对齐，将 CLIP中的知识提炼到我们的模型中
<center><img src=../images/image-40.png style="zoom:50%"></center>