# Open-Vocabulary DETR with Conditional Matching
https://arxiv.org/pdf/2203.11876
# Abstract
- DETR转变为开放词汇检测的最大挑战：
  - 如果不获得novel类别的标注图像，就不可能算出novel类别的分类成本矩阵
  - 为克服该挑战，本文将学习目标设定为输入queries (类名称或示例图像) 和对应目标之间的二分匹配，它会学习有用的对应关系以泛化到测试期间未见过的queries
- 提出基于DETR的开放词汇检测器OV-DETR：
  - 可以检测给定类名或示例图像的任何目标
  - 训练时，扩展了DETR的decoder来接受输入条件queries，本文根据从预训练的视觉语言模型如CLIP获得的输入embedding来影响transformer的decoder，以便能够匹配文本和图像queries
# Introduction
- 现有工作：
  - 主要方法：如ViLD，中心思想是将检测器的特征与在CLIP模型提供的嵌入对齐（见图 1 (a)）。这样，就可以使用对齐的分类器仅从描述性文本中识别新类别
  - 存在问题：现有的依赖于RPN，生成的区域提议会不可靠地覆盖图像中地所有类别，容易过拟合，无法推广到新类别
- 本文方法：
  - 提出训练一个end-to-end地开放词汇检测器，目的是不使用中间的RPN来提高novel类别的泛化性
  - 困难：
<center><img src=../images/image-50.png style="zoom:50%"></center>

# Related Work
- Visual Grounding
  - 视觉Grounding任务是使用自然语言输入在一张图像中ground住一个目标
  - 视觉Grounding方法通常涉及特定的单个目标，因此不能直接应用于通用目标检测
  - 相关工作：MDETR。
    - 该方法类似地使用给定的语言模型来训练DETR，以便将DETR的输出标记与特定单词链接起来。 MDETR还采用了条件框架，将视觉和文本特征结合起来馈送到encoder和decoder
    - 然而，该方法不适用于开放词汇检测，因为它无法计算分类框架下新类的成本矩阵

# Open-vocabulary DETR
动机：
- 将具有Closed-set匹配的标准DETR改造为需要与未见过的类进行匹配的开放词汇检测器并非易事
- 这种开放集匹配的一种直观方法是学习一个类别无关的模块（例如ViLD）来处理所有类别