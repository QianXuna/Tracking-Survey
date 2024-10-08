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

# Tracking-Every-Thing Tracker
模型框架如图5所示
<center><img src=../images/image-84.png style="zoom:50%"></center>

## Class-Agnostic Localization (CAL)
- 当普通目标检测器将定位和分类解耦时，发现检测器仍能定位罕见甚至novel的目标，图6显示了在TAO验证集上考虑和不考虑分类的目标检测器效果，不考虑类别预测时，检测器性能在罕见、常见和频繁类别中都很稳定，这说明检测器性能的瓶颈在分类器上
- 由于上边的分析，将常用的使用类别置信度的类内NMS替换为与类无关的对应方法
    <center><img src=../images/image-85.png style="zoom:50%"></center>

## Associating Every Thing
- 不同类别的运动线索是不规则的，所以使用外观线索
  - 不相信目标检测器的类别预测并将其作为硬先验，而是通过直接对比不同类别的样例来学习类别示例
  - 在关联过程中，使用类别示例来确定每个目标的潜在匹配候选目标。这一过程可以看作是将类别信息作为软先验信息来使用。因此，它可以整合分类所需的细粒度线索（例如，红色大巴士和红色卡车之间的区别）

## Class Exemplar Matching (CEM)
训练pipeline基于两阶段检测器，如图7所示，具体如下：
- RPN计算输入图片的所有RoI提议
- 使用RoI align从多尺度特征输出中提取特征图
- 特征图输入到exemplar encoder中来学习类别相似度，exemplar encoder为每个RoI生成类别示例
- 用定位阈值 $\alpha$ 为每个RoI分配类别标签
  - 如果一个RoI和一个Ground truth的IoU高于 $\alpha$ ，就给RoI分配对应的类别标签
  - 正样本是来自同一类别的RoI，负样本是来自不同类别的RoI
  - 修改SupCon loss，并提出一个不平衡的监督对比损失 (U-SupCon)：
    <center><img src=../images/image-87.png style="zoom:50%"></center>
<center><img src=../images/image-86.png style="zoom:50%"></center>