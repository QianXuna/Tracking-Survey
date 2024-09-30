# Global Tracking Transformers
https://arxiv.org/pdf/2203.13250
- 提出1个用于global MOT的transformer-based架构
  - 输入一段短的frames，然后为所有目标产生全局轨迹
  - transformer编码所有frame的目标特征，并使用轨迹queries将他们分组为轨迹，轨迹queries是来自单帧的对象特征，自然会产生独特的轨迹
  - 全局跟踪transformer不需要中间的成对分组或组合关联，并且可以与目标检测器联合训练
- MOT17: competitive performance
- TAO: SOTA
- 不是端到端结构、而且文章叙述乱且不清楚