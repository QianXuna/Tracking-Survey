# MotionTrack: Learning Robust Short-term and Long-term Motions for Multi-Object Tracking
## Abstract
- 提出了一个简单高效的 MOT 框架：MotionTrack
  - 用一个统一的框架学习短期和长期的运动，从而关联短期和长期的track
  - 对于密集人群：设计了一个Interaction Module，从短期轨迹中学习交互感知运动，它可以估计每个目标的复杂运动
  - 对于极度遮挡：设计了一个Refind Module，从目标的历史轨迹中学习可靠的长期运动，该模块可以将中断的轨迹与其相应的检测联系起来
- Interaction Module和Refind Module被嵌入到了tracking-by-detection范式中，它们可以协同工作以保持卓越的性能
- MOT17、MOT20数据集上达到了SOTA
- 应对场景：密集人群、极度遮挡

## Introduction
现存的MOT方法的两种范式：
- tracking-by-detection：首先检测每个视频帧中的对象，然后关联相邻帧之间的检测以随着时间的推移创建单独的目标轨迹
- tracking-by-regression：目标检测器不仅提供逐帧检测，还用每个轨迹到其新位置的连续回归来替换数据关联

面临的问题：短期和长期
- 短期：如何在短时间内将活动的轨迹与检测关联起来
  - 过去的方法：相邻帧使用可区分性的运动模式或外观特征来引导数据关联
  - 存在的问题：
    - 密集人群：行人的运动不是独立的，会受到周围邻居的影响以避免碰撞，故难以学习运动模式
    - 极度遮挡：行人很容易被固定物体长期遮挡，被遮挡的行人的检测框的尺寸会小到难以取得准确的外观特征
- 长期：如何在长时间后重新识别丢失的轨迹与检测
  - 过去的方法：
    - 学习可区分性的外观特征对被遮挡而丢失的track做重识别
    - memory技术：为每个目标存储特征以multi-query的方式匹配不同目标
  - 存在的问题：
    - 不同姿势、低分辨率、低照明：影响外观特征
    - memory模块和Multi-query机制的存储和时间开销很大：不利于实时tracking

贡献：
- 提出了一个简单高效的 MOT 框架：MotionTrack。遵循tracking-by-detection范式
- 为解决短期关联：设计了一个Interaction Module，用于模拟目标之间的所有交互
  - 可以预测行人的复杂运动以避免碰撞
  - 使用非对称邻接矩阵来表示目标之间的交互
  - 通过图卷积网络进行信息融合后得到预测
- 为解决长期关联：设计了一个Refind Module，用于重新识别丢失的目标
  - 相关性计算：将历史轨迹和当前检测的特征作为输入，并计算相关矩阵来表示它们关联的可能性
  - 误差补偿：修正被遮挡的轨迹
- MOT17、MOT20数据集上达到了 SOTA
