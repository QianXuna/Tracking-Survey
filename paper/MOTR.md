# MOTR: End-to-End Multiple-Object Tracking with TRansformer
https://arxiv.org/pdf/2105.03247

- 笔记：https://blog.csdn.net/wjpwjpwjp0831/article/details/121150712
- 关于detect queries如何区分是否new-born目标的问题：
  - github issue： https://github.com/megvii-research/MOTR/issues/46 （作者所解释的Deformable DETR具有重复删除机制不清楚）
  - **个人理解：对于DETR的decoder，将track queries放在detect queries左边后连接为一个序列，然后输入到masked self-attention层，track queries自然地去检测tracked object，位于序列右边的detect queries能看到左边的track queries，即已知tracked object，所以detect queries只会检测new-born目标**
  - 原文解释：MOTR Page5
    > Detect queries will only detect newborn objects since query interaction by self-attention in the Transformer decoder will suppress(抑制) detect queries that detect tracked objects. This mechanism is similar to duplicate removal in DETR that duplicate boxes are suppressed with low scores.
- 关于MOTR如何使用bbox初始化object的问题：https://github.com/megvii-research/MOTR/issues/21#issuecomment-957140009