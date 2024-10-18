# Tracking Every Thing in the Wild
https://arxiv.org/pdf/2207.12978
# Abstract
- 提出一个新的指标Track Every Thing Accuracy (TETA)
  - 将tracking分为3个子要素：定位、关联和分类
  - 该指标在分类不准确的情况下，也能对tracking性能进行全面的基准测试
  - 该指标也能处理大规模tracking数据集的不完整标注问题
- 提出Track Every Thing tracker (TETer) 
  - 使用类别示例匹配 (Class Exemplar Matching, CEM) 来进行关联
  - BDD100K、TAO：SOTA
# Instruction
- Association-Every-Thing (AET) 策略：不同于以前的方法只关联相同的类别，而是将相邻帧中的每个目标关联起来，将关联从大规模长尾条件下具有挑战性的分类/检测问题中解放出来。
  - 难点：关联过程中完全忽略了类别信息
  - 本文方法：引入类别样本匹配Class Exemplar Matching (CEM)
    - CEM学习的类别样本以soft方式合并有价值的类别信息
    - CEM可以有效利用大规模检测数据集的语义监督，不依赖于常常不正确的分类输出


# Tracking-Every-Thing Tracker
模型框架如图5所示
<center><img src=../images/image-84.png style="zoom:50%"></center>

## Class-Agnostic Localization (CAL)
- 当普通目标检测器将定位和分类解耦时，发现检测器仍能定位罕见甚至novel的目标，图6显示了在TAO验证集上考虑和不考虑分类的目标检测器效果，不考虑类别预测时，检测器性能在罕见、常见和频繁类别中都很稳定，这说明检测器性能的瓶颈在分类器上
- 由于上边的分析，将常用的使用类别置信度的类内NMS替换为与类无关的对应方法
    <center><img src=../images/image-85.png style="zoom:50%"></center>

## Associating Every Thing
不同类别的运动线索是不规则的，所以使用外观线索
- 不相信目标检测器的类别预测并将其作为硬先验，而是通过直接对比不同类别的样例来学习类别示例
- 在关联过程中，使用类别示例来确定每个目标的潜在匹配候选目标。这一过程可以看作是将类别信息作为软先验信息来使用。因此，它可以整合分类所需的细粒度线索（例如，红色大巴士和红色卡车之间的区别）

## Class Exemplar Matching (CEM)
训练pipeline基于两阶段检测器，如图7所示，具体如下：
- RPN计算输入图片的所有RoI proposals
- 使用RoI align从多尺度特征输出中提取特征图
- 特征图输入到exemplar encoder中来学习类别相似度，exemplar encoder为每个RoI生成类别exemplar
- 用定位阈值 $\alpha$ 为每个RoI分配类别标签
  - 如果一个RoI和一个Ground truth的IoU高于 $\alpha$ ，就给RoI分配对应的类别标签
  - 正样本是来自同一类别的RoI，负样本是来自不同类别的RoI
  - 修改SupCon loss，并提出一个不平衡的监督对比损失 (U-SupCon)：
    <center><img src=../images/image-87.png style="zoom:50%"></center>
    <center><img src=../images/image-93.png style="zoom:50%"></center>
<center><img src=../images/image-86.png style="zoom:50%"></center>

## Association Strategy
- 假设在第t帧，query目标q有类别examplar $q_c$， 在第t+1帧有检测出的目标D和它们的类别exemplars的集合 $d_c \in D_c$ 
- 计算 $q_c$ 和 $D_c$ 之间的相似度，并筛选高相似度的候选者，得到一个候选者列表
- 要从候选者列表中确定最终匹配结果，可以使用任何现有的关联方法
  - 在最终模型TETer中，进一步利用准密集相似性学习（quasi-dense similarity learning）来学习实例级关联的实例特征
  - 使用双向softmax和余弦相似性计算 C 中每个候选者的实例级匹配得分。选取得分最大的候选者

## Temporal Class Correction (TCC)
AET 策略允许我们利用丰富的时间信息来修正分类。如果我们跟踪一个物体，我们假定其类别标签在整个跟踪过程中都是一致的。我们使用简单的多数票来修正每帧的类别预测

# Appendix
## Exemplar-based Classification
给定一个示例目标，exemplar-based的分类是指通过与给定示例进行比较来确定目标是否属于同一类，从而对目标进行分类。给定视频序列中的两个相邻帧 t1 和 t2，t1 中的所有目标都将被视为exemplar。对于每个exemplar，我们找到 t2 中与该exemplar属于同一类的所有目标
## More Implementation Details
### Network architecture
- 检测器：ResNet的Faster R-CNN
- exemplar encoder: 4conv-3fc head + group normalization

### Train: TAO
- 检测器的训练：LVISv0.5、COCO数据集
- exemplar encoder的训练：TAO训练集

### Train: BDD100K
- 检测器的训练：following QDTrack which follows ByteTrack
- exemplar encoder的训练：8个类别的BDD 100K训练集

### Test: TAO
- Eval: TAO val set + TETA
  

问题：
- TETer的训练集的类别到底是否有novel类别？
  - OVTrack说TETer的训练集是base和novel类别之和
