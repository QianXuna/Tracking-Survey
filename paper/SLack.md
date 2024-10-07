# SLAck: Semantic, Location, and Appearance Aware Open-Vocabulary Tracking
https://arxiv.org/pdf/2409.11235  
# Abstract
- 以往的方法基于纯外观来匹配，由于开放词汇中运动模式的复杂性和novel目标分类的不稳定性，现有方法在最后匹配阶段，要么忽略运动和语义线索，要么根据启发式方法进行匹配
- 本文提出了一个联合的框架SLAck，在关联的早期阶段连接了语义、位置和外观先验，并通过一个轻量级的时间和空间目标图来学习如何联合所有有价值的信息
  - 本文方法无需使用复杂的后处理启发式方法来融合不同的线索，从而大大提高了大规模开放词汇跟踪的关联性能
  - 在MOT和TAO的TETA基准上达到了SOTA

