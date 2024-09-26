# TAO: A Large-Scale Benchmark for Tracking Any Object
https://arxiv.org/pdf/2005.10356
# Abstract
- 提出Tracking Any Object (TAO)数据集
  - 2907个高分辨率视频，平均长度为1分钟
  - 833个类别的词汇表：要求标注者标记视频中任意点移动的目标，并事后为它们命名
  - 为了确保标注的可扩展性，本文采用了一种联合方法，该方法将手动的工作重点放在视频中相关目标（例如移动的目标）的track的标注上
- 在开放世界的大词汇量跟踪上，评估了最先进的跟踪器
  - 现有的SOT、MOT跟踪器应用这种场景会有困难
  - detection-based的MOT跟踪器比user-initialized的跟踪器更具有竞争力

# Introduction
现有的检测数据集规模和种类都很大，相反地，MOT数据集很小，如表1所示，从图1可知，MOT数据集的类别主要是人和车辆
<center><img src=../images/image-32.png style="zoom:50%"></center>
<center><img src=../images/image-33.png style="zoom:50%"></center>