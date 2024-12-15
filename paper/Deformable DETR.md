# Deformable DETR: Deformable Transformers for End-to-End Object Detection
https://arxiv.org/pdf/2010.04159  
很棒的解析：https://zhuanlan.zhihu.com/p/372116181
# ABSTRACT
- DETR的问题：由于Transformer注意力模块在处理图像特征图方面的限制，它存在收敛速度慢和特征空间分辨率有限的问题
- 提出Deformable DETR
  - 其注意力模块仅关注reference点周围的一小组关键采样点
  - 提出可变形注意力模块，关注一小组采样位置作为所有特征图像素中关键元素的预过滤器
  - 该模块可以自然扩展到聚合多尺度特转增，而无需FPN
- 在小目标上比DETR有更好地性能，缓解DETR收敛速度慢和复杂度高的问题，训练次数减少10倍
- COCO

# INTRODUCTION
- DETR的问题：
  - 1. 相比现存的目标检测器，DETR需要更长的训练epochs去收敛
  - 2. DETR在检测小目标上表现出较低的性能水平，现代目标检测器通常会利用多尺度特征，从高分辨率的特征图上检测小目标。同时，高分辨率特征图会为DETR带来不可接受的复杂度
- 上述问题归因：
  - 1. Transformer组件在处理图像特征图方面的不足。在初始化时，注意力模块将几乎统一的注意力权重投射到特征图中的所有像素。长时间的训练对于学习注意力权重以关注稀疏的有意义的位置是必要的。
  - 2. Transformer encoder中注意力权重计算是关于像素数的二次计算。此，处理高分辨率特征图的计算和存储复杂度非常高。
- 本文贡献：提出Deformable DETR，缓解了DETR收敛慢和复杂度高的问题
  - 1. 联合了deformable convolution的稀疏空间采样和Transformers的关系建模能力的优点，提出了deformable attention模块，该模块关注一小组采样位置，作为所有特征图像素中突出的关键元素 (prominent key elements) 的预过滤器 (pre-filter)。该模块还能自然地扩展到聚合多尺度特征，并且不需要FPN的帮助，我们利用deformable attention模块代替了transformer attention模块来处理特征图。
  - 2. 由于Deformable DETR快速的收敛性以及计算和内存的效率，开发其变体成为可能。
    - 我们开发了一个简单且高效的 iterative bounding box refinement 机制来提高检测性能。
    - 我们尝试了一个变体，叫做two-stage Deformable DETR，该模型能够产生region proposals，而且这些region proposals进一步被输入到decoder中来进行 iterative bounding box refinement


Deformable DETR如图1所示：
<center><img src=../images/image-131.png style="zoom:50%"></center>

# REVISITING TRANSFORMERS AND DETR
## Multi-Head Attention in Transformers
多头注意力结构：
<center><img src=../images/image-149.png style="zoom:50%"></center>
<center><img src=../images/image-150.png style="zoom:50%"></center>

Transformer的问题：
<center><img src=../images/image-148.png style="zoom:50%"></center>

- 注意力权重的均值为0，指数的零次方为1，共有 $N_k$ 个注意力权重，所以注意力权重约等于 $\frac{1}{Nk}$

## DETR
DETR结构：
<center><img src=../images/image-151.png style="zoom:50%"></center>
DETR的问题：
<center><img src=../images/image-152.png style="zoom:50%"></center>

<center><img src=../images/image-137.png style="zoom:50%"></center>

# METHOD
## DEFORMABLE TRANSFORMERS FOR END-TO-END OBJECT DETECTION
### Deformable Attention Module
- Transformer attention应用在图像特征图上的核心问题是它会看所有可能的空间位置。为解决该问题，我们提出可变形注意力(deformable attention) 模块。可变形注意力模块仅关注参考点 (reference points) 周围的一小组关键采样点 (key sampling points)，而不管特征图的空间大小，**结构如图2所示**。通过为每个query仅分配少量固定的key，可以缓解收敛和特征空间分辨率问题。
- 给定输入特征图 $x \in R^{C \times H \times W}$ ，令 q 索引包含了内容特征 $z_{q}$ 和 2d 参考点 $p_q$ 的query元素，可变形注意力特征用下式计算：  
<center><img src=../images/image-165.png style="zoom:70%"></center>

- 其中，m为attention head的索引，k为采样的keys的索引，K为总计采样key的个数 (K << HW) 。 $\Delta p_{mqk}$ 和 $A_{mqk}$ 分别代表第m个attention head中第k个采样点的采样偏移量 (sampling offset) 和注意力权重。$\Delta p_{mqk}$ 为无限制的2-d实数， $A_{mqk}$ 为0-1之间的数。$\Delta p_{mqk}$ 和 $A_{mqk}$ 通过对 query 特征 $z_{q}$ 做线性投影得到。在实现时， $z_q$ 被输入到包含 3MK 个通道的线性投影操作中，其中前 2MK 个通道编码  (encode) 采样偏移 $\Delta p_{mqk}$ ，剩下的 MK 个通道输入到 softmax 操作中来获得注意力权重 $A_{mqk}$。

- <u>问题：为什么注意力权重 $A_{mqk}$ 可以由全连接层直接预测得到？</u>
- 原因：打个比方：A与B已建立起了对应关系，之后A再通过某种映射关系得到C，B也通过某种映射关系得到D，那么C与D之间必然会有某种程度的耦合与对应关系。这里A、B、C、D就分别指代query、reference points、attention weights以及value。【参考：https://zhuanlan.zhihu.com/p/372116181】

<center><img src=../images/image-138.png style="zoom:70%"></center>

- 复杂度：
<center><img src=../images/image-140.png style="zoom:50%"></center>
<center><img src=../images/image-141.png style="zoom:50%"></center>


### Multi-scale Deformable Attention Module
大多数现代目标检测框架受益于多尺度特征图，本文提出的可变形注意力模块能够自然地扩展到多尺度特征图。多尺度注意力模块的公式如下所示：
<center><img src=../images/image-142.png style="zoom:50%"></center>

- 当L=1，K=1且 $W_m'$ 固定为单位矩阵时，可变形注意力模块退化为可变形卷积模块，可变形卷积为单尺度特征图输入设计的，每个注意力头仅关注一个采样点
- 本文的多尺度可变形注意力会关注来自多尺度输入的多个采样点
- 所提出的（多尺度）可变形注意模块也可以被视为 Transformer 注意力有效变体，其中通过可变形采样位置引入预过滤机制。当采样点遍历所有可能的位置时，所提出的注意力模块相当于Transformer注意力
- <u>问题：为何不需要FPN也能达到跨层融合的效果？</u>
- 回答：https://zhuanlan.zhihu.com/p/372116181

### Deformable Transformer Encoder
我们用多尺度可变形注意力模块MSDeformAttn取代DETR的Transformer注意力模块MultiHeadAttn，具体而言：
- 多尺度特征图的构建：
  - 从ResNet的 $C_3$ 到 $C_5$ 阶段的输出特征图中提取多尺度特征图 $\{x^l\}_{l=1}^{L-1} (L=4)$ (通过1*1卷积)，其中 $C_l$ 的分辨率比输入图像低 $2^l$ 倍
  - 最低分辨率特征图 $x^L$ 通过 $C_5$ 阶段的 3*3 stride=2 的卷积获得，表示为 $C_6$
  - 所有多尺度特征图有C=256个通道
  - 没有使用FPN中的自上而下结构，因为根据式(3)可以看到，多尺度可变形注意力本身可以在多尺度特征图之间交换信息
- encoder的结构：
  - 输入和输出都是有相同分辨率的多尺度特征图， key 元素和 query 元素都是多尺度特征图中的元素，对于每个 query 像素，参考点是它本身
  - 为了识别每个 query 像素所处的特征层，除了位置特征向量外，还在特征表征中添加了一个尺度级特征向量 $e_l$ ，不同于固定编码的位置向量，尺度级向量 $\{e_l\}_{l=1}^L$ 为随机初始化的，且与网络共同训练
- **结构如附录A.2**：
    <center><img src=../images/image-143.png style="zoom:50%"></center>

### Deformable Transformer Decoder
- decoder中包括cross-attention和self-Attention，两种注意类模块的query元素都是object query，在cross-attention模块中，object query从特征图中提取特征，其中 key 元素是encoder的输出特征图。在self-attention中，object query相互交互，其中 key 元素是object query。由于可变形注意力模块是针对特征图作为 key 元素而设计的，因此我们只将cross-attention替换为多尺度可变形注意力，保持self-attention不变
- **参考点 $\hat{p_q}$ 的2d标准化的坐标是由object query embedding通过一个可学习的线性投影层+sigmoid函数预测到的**
- **参考点被用作边界框中心的初始猜测**
- 由于多尺度可变形注意力模块提取的是参考点周围的图像特征，因此让检测头预测边界框与参考点的**相对偏移量 (relative offsets)**，以进一步降低优化难度，**详见附录A.3**。
  - 问题：为何检测头部的回归分支预测的是偏移量而非绝对坐标值？
  - 原因：采样点的位置是基于参考点和对应的坐标偏移量计算出来的，也就是说采样特征是分布在参考点附近的，既然这里需要由采样特征回归出bbox的位置，那么预测相对于参考点的偏移量就会比直接预测绝对坐标更易优化，更有利于模型学习。【参考：https://zhuanlan.zhihu.com/p/372116181】
- 作用：学习到的decoder注意力将与预测的边界框有很强的相关性
  <center><img src=../images/image-144.png style="zoom:50%"></center>

## ADDITIONAL IMPROVEMENTS AND VARIANTS FOR DEFORMABLE DETR
可变形DETR为利用端到端目标检测器的各种变体提供了可能性，**详见附录A.4**，现在介绍这些改进和变体：
### Iterative Bounding Box Refinement
参考论文Raft: Recurrent all-pairs field transforms for optical flow，本文设计了一个简单、高效的迭代边界框细化机制来提高检测性能，每个decoder layer根据前一层的预测来细化边界框。每层Decoder都是可以输出bbox和分类信息的，如果都利用起来算损失则成为auxiliary loss。具体描述如下：
<center><img src=../images/image-145.png style="zoom:70%"></center>
<center><img src=../images/image-146.png style="zoom:70%"></center>

仅当使用了iterative bbox refine策略时有这一步：使用bbox检测头部对解码特征进行预测，得到相对于参考点(boxes or points)的偏移量，然后加上参考点坐标（先经过反sigmoid处理，即先从归一化的空间从还原出来），最后这个结果再经过sigmoid（归一化）得到校正的参考点，供下一层使用（在输入下一层之前会取消梯度，因为这个参考点在各层相当于作为先验的角色）【参考：https://zhuanlan.zhihu.com/p/372116181】

### Two-Stage Deformable DETR
原始DETR中，decoder中的object queries与当前图像无关，受到两阶段目标检测器的启发 (如RCNN，即先提取候选区域，再对区域分类)，本文探索了可变形DETR的变体，即再第一阶段生成候选区域，生成的候选区域会被作为object queries输入decoder以进一步细化，形成两阶段可变形DETR  
在第一阶段，为了实现高召回率候选框，多尺度特征图中的每个像素将充当object query。然而，直接将object queries设置为像素会给decoder中的self-attention模块带来不可接受的计算和内存成本，其复杂性随着查询数量呈二次方增长。为了避免这个问题，本方法删除了decoder并形成一个encoder-only的 Deformable DETR 来生成区域候选框。其中，每个像素都被分配为一个object query，它直接预测一个边界框。得分最高的边界框被认为是候选区域。在将候选区域馈送到第二阶段之前，不应用 NMS
具体描述如下：
<center><img src=../images/image-147.png style="zoom:70%"></center>

TODO：
- 把所有的细节画一幅图，再给老师讲