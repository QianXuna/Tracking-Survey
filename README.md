"# Tracking-Survey"
repo 列：

- 无：无仓库
- not release：有仓库但未开源

# 论文列表

|列表名|repo|year（知识库为更新时间）|观点|
|-----|----|-----------------------|-----|
|知识库|[awesome-multiple-object-tracking](https://github.com/luanshiyinyang/awesome-multiple-object-tracking)|2024|常规多目标跟踪|
|知识库（<https://arxiv.org/abs/2405.14200）>|[Awesome-Multimodal-Object-Tracking](https://github.com/983632847/Awesome-Multimodal-Object-Tracking)|2024|多模态多目标跟踪（MMOT），包括RGBD、RGBT、RGBE、RGBL、混合模态tracking|
|知识库|[awesome-3d-multi-object-tracking-autonomous-driving](https://github.com/MagicTZ/awesome-3d-multi-object-tracking-autonomous-driving)|2022|自动驾驶3D多目标跟踪|
|知识库|[awesome-3d-multi-object-tracking](https://github.com/Po-Jen/awesome-3d-multi-object-tracking)|2022|3D多目标跟踪|
|[ICCV 2023](https://openaccess.thecvf.com/ICCV2023?day=all)|<https://blog.csdn.net/CV_Autobot/article/details/132222653> <https://github.com/amusi/ICCV2023-Papers-with-Code?tab=readme-ov-file#VThttps://github.com/amusi/ICCV2023-Papers-with-Code?tab=readme-ov-file#VT>|2023||
|[CVPR 2023](https://openaccess.thecvf.com/CVPR2023?day=all)|<https://zhuanlan.zhihu.com/p/615368672>|2023|
|[NIPS 2023](https://proceedings.neurips.cc/paper/2023)|<https://openreview.net/group?id=NeurIPS.cc/2023/Conference#tab-accept-oral>|2023||
|[CVPR 2024](https://openaccess.thecvf.com/CVPR2024?day=all)|<https://github.com/52CV/CVPR-2024-Papers?tab=readme-ov-file#25>|2024||
|AAAI 2024|<https://github.com/DmitryRyumin/AAAI-2024-Papers?tab=readme-ov-file>|2024||
|[ECCV 2024](https://eccv2024.ecva.net/virtual/2024/papers.html?filter=titles)|<https://github.com/amusi/ECCV2024-Papers-with-Code>|2024||

# Paper
## 综述
### Open-vocabulary learning
| paper | repo | year| jyliu 观点 | kgmao 观点|
| ----- | ---- | --- | ---------- | --------- |
|[Towards Open Vocabulary Learning: A Survey](https://arxiv.org/pdf/2306.15880)|无|TPAMI 2024||
|[A Survey on Open-Vocabulary Detection and Segmentation: Past, Present, and Future](https://arxiv.org/pdf/2307.09220)|[awesome](https://github.com/seanzhuh/awesome-open-vocabulary-detection-and-segmentation)|TPAMI 2024||

## 检测
### Open Vocabulary OD
| paper | repo | year| jyliu 观点 | kgmao 观点|
| ----- | ---- | --- | ---------- | --------- |
|[ViLD](https://arxiv.org/pdf/2104.13921)|[repo](https://github.com/jhoowy/ViLD)|ICLR 2022|CLIP、zero-shot detection|
|[DetPro](https://arxiv.org/pdf/2203.14940)|[repo](https://github.com/dyabel/detpro)|CVPR 2022||
|[DK-DETR](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Distilling_DETR_with_Visual-Linguistic_Knowledge_for_Open-Vocabulary_Object_Detection_ICCV_2023_paper.pdf)|[repo](https://github.com/hikvision-research/opera?tab=readme-ov-file)|ICCV 2023|DETR、CLIP|
|[SIC-CADS](https://arxiv.org/pdf/2312.10439)|[repo](https://github.com/mala-lab/SIC-CADS)|AAAI 2024|1. 发现并定义了共生和相互依赖这两种潜藏在相似场景下不同类别之间的关系，传统的基于区域概念的学习方法不能够有效检测某个外观较小、模糊或被遮挡的目标 （称为hard objects）<br> 2. 所提出的方法是一个简单、轻量级的通用框架，可以轻松插入不同的现有 OVOD 模型，以增强其检测hard objects的能力|
|[YOLO-World](https://openaccess.thecvf.com/content/CVPR2024/papers/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.pdf)|[repo](https://github.com/AILab-CVC/YOLO-World)|CVPR 2024||
|[HyperLearner](https://openaccess.thecvf.com/content/CVPR2024/html/Kong_Hyperbolic_Learning_with_Synthetic_Captions_for_Open-World_Detection_CVPR_2024_paper.html)|无|CVPR 2024||
|[BIND](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Exploring_Region-Word_Alignment_in_Built-in_Detector_for_Open-Vocabulary_Object_Detection_CVPR_2024_paper.html)|无|CVPR 2024||
|[SHiNe](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_SHiNe_Semantic_Hierarchy_Nexus_for_Open-vocabulary_Object_Detection_CVPR_2024_paper.html)|[repo](https://github.com/naver/shine)|CVPR 2024||
|[DetCLIPv3](https://openaccess.thecvf.com/content/CVPR2024/html/Yao_DetCLIPv3_Towards_Versatile_Generative_Open-vocabulary_Object_Detection_CVPR_2024_paper.html)|无|CVPR 2024||
|[LBP](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Learning_Background_Prompts_to_Discover_Implicit_Knowledge_for_Open_Vocabulary_CVPR_2024_paper.html)|无|CVPR 2024||
|[SAMP](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_Scene-adaptive_and_Region-aware_Multi-modal_Prompt_for_Open_Vocabulary_Object_Detection_CVPR_2024_paper.html)|无|CVPR 2024||
|[SAS-Det](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_Taming_Self-Training_for_Open-Vocabulary_Object_Detection_CVPR_2024_paper.html)|[repo](https://github.com/xiaofeng94/SAS-Det)|CVPR 2024|self-training|
|[RALF](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_Retrieval-Augmented_Open-Vocabulary_Object_Detection_CVPR_2024_paper.html)|[repo](https://github.com/mlvlab/RALF)|CVPR 2024||
|[MarvelOVD](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02551.pdf)|[repo](https://github.com/wkfdb/MarvelOVD)|ECCV 2024||
|[CLIFF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07221.pdf)|[repo](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07221.pdf)|ECCV 2024||
[Grounding DINO](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06319.pdf)|[repo](https://arxiv.org/pdf/2303.05499)|ECCV 2024||
|[OpenSight](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11118.pdf)|not release|ECCV 2024|3D|

### Open World OD
| paper | repo | year| jyliu 观点 | kgmao 观点|
| ----- | ---- | --- | ---------- | --------- |
|[OWOD](https://openaccess.thecvf.com/content/CVPR2021/html/Joseph_Towards_Open_World_Object_Detection_CVPR_2021_paper.html)|[repo](https://github.com/JosephKJ/OWOD)|CVPR 2021 oral|
|[OV-DETR](https://arxiv.org/pdf/2203.11876)|[repo](https://github.com/yuhangzang/OV-DETR)|ECCV 2022|DETR、用CLIP的图像/文本特征改造object query|

### Common OD
| paper | repo | year| jyliu 观点 | kgmao 观点|
| ----- | ---- | --- | ---------- | --------- |
|[DETR](https://arxiv.org/pdf/2005.12872)|[repo](https://github.com/facebookresearch/detr)|2020 arxiv|||
|[Deformable DETR](https://arxiv.org/pdf/2010.04159)|[repo](https://github.com/fundamentalvision/Deformable-DETR)|ICLR 2021 oral|||


## 多目标
### Open Vocabulary MOT
| paper | repo | year|jyliu 观点|kgmao 观点|
| ----- | ---- | --- | -------- | ---------|
|[TAO](https://arxiv.org/pdf/2005.10356)|[dataset](https://taodataset.org/)|ECCV 2020|跟踪模型的调参方法：基于MAP指标|
|[AOA](https://arxiv.org/pdf/2101.08040)|[repo](https://github.com/feiaxyt/Winner_ECCV20_TAO)|ECCV-TAO-2020|集成几个模型|
|[QDTrack](https://arxiv.org/pdf/2006.06664)|[repo](https://github.com/SysCV/qdtrack)|CVPR 2021 oral|1. 对一对图像上的数百个区域进行密集采样来进行对比学习，图像对是视频的某一帧及其在时间领域的随机采样，区域的密集采样使用RPN，特征提取使用外观线索<br>2. 关联算法是双向softmax，当前帧的检测目标和过去x帧的候选目标之间做双向softmax，高分表明两个匹配目标在特征空间中彼此最近，可以使用最近邻搜索将目标关联起来|
|[QDTrack](https://ieeexplore.ieee.org/document/10209207)|[repo](https://github.com/SysCV/qdtrack)|TPAMI 2023|1. 对一对图像上的数百个区域进行密集采样来进行对比学习，图像对是视频的某一帧及其在时间领域的随机采样，区域的密集采样使用RPN，特征提取使用外观线索，与2021的QDTrack的框架相同<br>2. 在2021QDTrack的基础上证明使用静态图像的数据增强能提高性能||
|[GTR](https://arxiv.org/pdf/2203.13250)|[repo](https://github.com/xingyizhou/GTR)|CVPR 2022|1. Transformer<br>2. MOT17、TAO<br>3. 不是端到端结构、而且文章叙述乱且不清楚|
|[TETer](https://arxiv.org/pdf/2207.12978)|[project](https://www.vis.xyz/pub/tet/)|ECCV 2022|1. TETA指标：定位、关联和分类<br>2. TETer跟踪器框架<br>3. benchmark：BDD100K、TAO|
|[OVTrack](https://arxiv.org/pdf/2304.08408)|[project](<https://www.vis.xyz/pub/ovtrack/>)|CVPR 2023|1. 定义Open-vocabulary MOT: 要求跟踪训练集的词汇表中未见过的种类<br>2. 用嵌入头替换检测器（Faster R-CNN）的分类头，通过将目标候选框的图像特征表征与相应的CLIP图像和文本嵌入对齐，学习出一个用于生成图像embedding的图像头和文本embedding的文本头，将CLIP中的知识蒸馏到我们的模型中<br>3. 用DDPM得到静态图像对，该对可以形成对检测器的跟踪头的监督，用有监督的对比损失函数来学习关联|
|[MASA](https://arxiv.org/pdf/2406.04221)|[project](https://matchinganything.github.io/)|CVPR 2024|1. 用数据增强构建两个不同视图，实现像素级对应关系，用SAM自动将相同实例的像素进行分组，生成实例级对应关系，即得到自监督信号，使用对比学习公式来学习<br>2. 设计了一个adapter，用于和检测 (如Detic、Grounding-DINO) 或分割 (如SAM) 基础模型集成，训练阶段：两种基础模型的主干是冻结的，分别设计对应的模型将基础模型的主干特征转换为适合跟踪的新特征，同时与SAM集成时，给adapter增加检测头，蒸馏SAM的检测能力到RCNN的检测头上，adapter的训练使用检测损失和对比损失之和。推理阶段：与目标检测器集成时，输出track features，与SAM集成时，adapter输出track features和检测框|
|[SLack](https://arxiv.org/pdf/2409.11235)|[repo](<https://github.com/siyuanliii/SLAck>)|ECCV 2024|1. Tracking-by-detection的范式，和TETer、OVTrack相同的目标检测器上构建跟踪器<br>2. 检测器上使用语义头、位置头、外观头，得到语义、位置、外观的embedding，将三者做加来融合<br>3. 在帧内对融合后的特征使用self-attention，在帧间使用cross-attention<br>4. 用ground truth构建匹配矩阵，使用Sinkhorn损失来更新softmax归一化的得分矩阵，这是个最优传输问题<br>5. 为了使用TAO数据集的稀疏标注，检测器首先推断训练视频上的边界框，在训练和测试阶段保持输入数据的一致性。仅在这些预测框与可用的ground truth之间存在匹配时计算关联损失，忽略不匹配的对||
|[GLATrack](https://openreview.net/pdf?id=ya9wqTWe7a)|not release|ACM MM 2024||

### Common MOT
| paper | repo | year|jyliu 观点|kgmao 观点|
| ----- | ---- | --- | -------- | ---------|
|[MOTR](https://arxiv.org/pdf/2105.03247)|[repo](https://github.com/megvii-research/MOTR)|ECCV 2022||
|[MotionTrack](https://arxiv.org/pdf/2303.10404)|无|CVPR 2023|1. Track-by-detection范式<br>2. 改进短期关联，提出新的运动模型：不用KF预测track的位置，用自注意力、卷积建立一个交互矩阵，该矩阵建模了一个track对另一个track的影响，对交互矩阵用图卷积和MLP，预测位置的偏移量<br>3. 改进长期关联，提出新的关联模型和误差补偿模型：对丢失track的时间分布模式和速度-时间关系建模，得到相关性矩阵，再做匹配；用匹配到的检测和丢失track的预测来推理遮挡期间的预测误差，并完善track的预测|
|[GHOST](https://arxiv.org/pdf/2206.04656)|[repo](<https://github.com/dvl-tum/GHOST>)|CVPR 2023|1. Track-by-detection范式<br>2. 改进reID模型：在直方图上发现了inactive和active的track与检测的外观距离之间的差异很大的同时有很大的重叠面积，在此启发下认为计算inactive track和active track应该分别采取不同的处理方式计算和检测的外观距离，对inactive track采用proxy distance，缩小了直方图上inactive和active的重叠面积<br>3. 改进reID模型：动态域适应，在常规的人群reID数据集上训好reID模型，把这个模型在MOT数据集的每个sequence上训练BN层，看起来似乎BN层的训练和tracking的推理是同时进行的|
|[OC-SORT](https://arxiv.org/pdf/2203.14360)|[repo](<https://github.com/noahcao/OC_SORT>)|CVPR 2023|1. 改进Kalman Filter。遮挡情况下，Kalman Filter只根据上一时刻的预测结果预测当前时刻的位置，遮挡时间越长，误差积累越多。用遮挡前和遮挡后的检测框位置生成目标被遮挡期间的虚拟轨迹，利用虚拟轨迹进行Kalman<br>2. 提出角度损失angle_diff_cost|
|[FineTrack](https://arxiv.org/pdf/2302.14589)|无|CVPR 2023|优化reID，从全局和局部等不同粒度描述外观|
|[RMOT、Refer-KITTI、TransRMOT](https://arxiv.org/pdf/2303.03366)|[repo](<https://github.com/wudongming97/RMOT>)<br>[supplementary material](<https://openaccess.thecvf.com/content/CVPR2023/supplemental/Wu_Referring_Multi-Object_Tracking_CVPR_2023_supplemental.pdf>)|CVPR 2023|1. 指代多目标跟踪 (RMOT) 任务：给定语言表达作为参考，它的目标是为定位视频中所有语义匹配的目标<br>2. Refer-KiTTI 数据集：标注成本低。提供了一个高效的标注工具<br>3. TransRMOT 框架：end-to-end 模型，类似MOTR，主要区别是扩展encoder为cross-modal encoder，损失为track loss+detect loss，每种loss由是否存在目标、目标框位置、是否为表达式指代的目标组成|
|[MOTRv2](https://arxiv.org/pdf/2211.09791)|[repo](<https://github.com/noahcao/OC_SORT>)|CVPR 2023|1. 引入 YOLOX 的输出作先验<br>2. proposal query ：用一个可学习的共享的 query 和 YOLOX 输出置信分数的加和<br>3. YOLOX proposals：YOLOX 输出的框坐标。当前帧的 YOLOX proposals 和 上一帧预测的框坐标 Yt-1 连接起来，作为 proposal query 的 anchor 点，两者之间做加和|
|[VoxelNeXt](https://arxiv.org/pdf/2303.11301)|[repo](<https://github.com/dvlab-research/VoxelNeXt>)|CVPR 2023|3D 检测、3D tracking|
|[PF-Track](https://arxiv.org/pdf/2302.03802)|[repo](<https://github.com/TRI-ML/PF-Track>)|CVPR 2023|3D、multi-camera|
|[UTM](https://openaccess.thecvf.com/content/CVPR2023/papers/You_UTM_A_Unified_Multiple_Object_Tracking_Model_With_Identity-Aware_Feature_CVPR_2023_paper.pdf)|无|CVPR 2023||
|[SUSHI](https://openaccess.thecvf.com/content/CVPR2023/papers/Cetintas_Unifying_Short_and_Long-Term_Tracking_With_Graph_Hierarchies_CVPR_2023_paper.pdf)|[repo](<https://github.com/dvl-tum/SUSHI>)|CVPR 2023||
|[DETracker](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Tracking_Multiple_Deformable_Objects_in_Egocentric_Videos_CVPR_2023_paper.pdf)|not release|CVPR 2023|DogThruGlasses 数据集、DETracker 框架||
|[ColTrack](https://arxiv.org/pdf/2308.05911)|[repo](<https://github.com/bytedance/ColTrack>)|ICCV 2023|低帧率|
|[MeMOTR](https://arxiv.org/pdf/2307.15700)|[repo](<https://github.com/MCG-NJU/MeMOTR>)|ICCV 2023||
|[TrackFlow](https://arxiv.org/pdf/2308.11513)|无|ICCV 2023||
|[ReST](https://arxiv.org/pdf/2308.13229)|[repo](<https://github.com/chengche6230/ReST>)|ICCV 2023|multi-camera|
|[FUS3D](https://openaccess.thecvf.com/content/ICCV2023/papers/Heitzinger_A_Fast_Unified_System_for_3D_Object_Detection_and_Tracking_ICCV_2023_paper.pdf)|[repo](<https://github.com/theitzin/FUS3D>)|ICCV 2023|3D 检测、3D tracking|
|[HD-AMOT](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Heterogeneous_Diversity_Driven_Active_Learning_for_Multi-Object_Tracking_ICCV_2023_paper.pdf)|无|ICCV 2023||
|[3DMOTFormer](https://arxiv.org/pdf/2308.06635)|[repo](<https://github.com/dsx0511/3DMOTFormer>)|ICCV 2023|3D|
|[OC-MOT](https://arxiv.org/pdf/2309.00233))|[repo](<https://github.com/amazon-science/object-centric-multiple-object-tracking>)|ICCV 2023|无监督|
|[TrajectoryFormer](https://arxiv.org/pdf/2306.05888)|[repo](<https://github.com/poodarchu/EFG>)|ICCV 2023|3D|
|[SportsMOT、MixSort](https://arxiv.org/pdf/2304.05170)|[project](<https://deeperaction.github.io/datasets/sportsmot.html>)|ICCV 2023|SportsMOT 数据集、MixSort 框架|
|[U2MOT](https://arxiv.org/pdf/2307.15409)|[repo](<https://github.com/alibaba/u2mot/>)|ICCV 2023|无监督|
|[DQTrack](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_End-to-end_3D_Tracking_with_Decoupled_Queries_ICCV_2023_paper.pdf)|[project](<https://sites.google.com/view/dqtrack>)|ICCV 2023|3D|
|[DARTH](https://openaccess.thecvf.com/content/ICCV2023/papers/Segu_DARTH_Holistic_Test-time_Adaptation_for_Multiple_Object_Tracking_ICCV_2023_paper.pdf)|[repo](<https://github.com/mattiasegu/darth>)|ICCV 2023|域偏移|
|[UCSL](https://arxiv.org/pdf/2309.00942)|无|ICCV 2023|无监督|
|[Type-to-Track](https://arxiv.org/pdf/2305.13495)|[repo](<https://github.com/uark-cviu/Type-to-Track>)|NIPS 2023|RMOT、GroOT数据集、MENDER框架|
|[Hybrid-SORT](https://browse.arxiv.org/pdf/2308.00783)|[repo](<https://github.com/ymzis69/HybridSORT>)|AAAI 2024||
|[UCMCTrack](https://browse.arxiv.org/pdf/2312.08952)|[repo](<https://github.com/corfyi/UCMCTrack>)|AAAI 2024||
|[SRTrack](https://ojs.aaai.org/index.php/AAAI/article/view/28115)|[repo](<https://github.com/lzzppp/SR-Track>)|AAAI 2024||
|[Multi-Scene Generalized Trajectory Global Graph Solver with Composite Nodes for Multiple Object Tracking](https://arxiv.org/pdf/2312.08951)|无|AAAI 2024||
|[DiffusionTrack](https://arxiv.org/pdf/2308.09905)|[repo](<https://github.com/RainBowLuoCS/DiffusionTrack>)|AAAI 2024||
|[SlowTrack](https://ojs.aaai.org/index.php/AAAI/article/view/28200)|无|AAAI 2024||
|[SMILEtrack](https://ojs.aaai.org/index.php/AAAI/article/view/28386)|[repo](<https://github.com/pingyang1117/SMILEtrack_Official>)|AAAI 2024||
|[NetTrack](https://arxiv.org/pdf/2403.11186v1)|[project](<https://george-zhuang.github.io/nettrack/>)|CVPR 2024||
|[LMOT](https://arxiv.org/pdf/2405.06600)|not release|CVPR 2024||
|[GeneralTrack](https://arxiv.org/pdf/2406.00429)|[repo](<https://github.com/qinzheng2000/GeneralTrack>)|CVPR 2024||
|[ADATrack](https://arxiv.org/pdf/2405.08909)|[repo](<https://github.com/dsx0511/ADA-Track>)|CVPR 2024|3D、multi-camera|
|[DeconfuseTrack](https://arxiv.org/pdf/2403.02767)|无|CVPR 2024||
|[Delving into the Trajectory Long-tail Distribution for Muti-object Tracking](https://arxiv.org/abs/2403.04700v2)|[repo](<https://github.com/chen-si-jia/Trajectory-Long-tail-Distribution-for-MOT>)|CVPR 2024||
|[Self-Supervised Multi-Object Tracking with Path Consistency](https://arxiv.org/pdf/2404.05136v1)|[repo](<https://github.com/amazon-science/path-consistency>)|CVPR 2024||
|[DiffMOT](https://arxiv.org/abs/2403.02075)|[project](<https://diffmot.github.io/>)|CVPR 2024||
|[iKUN](https://arxiv.org/pdf/2312.16245)|[repo](<https://github.com/dyhBUPT/iKUN>)|CVPR 2024||
|[DepthMOT](https://arxiv.org/pdf/2404.05518)|[repo](<https://github.com/JackWoo0831/DepthMOT>)|ECCV 2024 (×)||
|[MOTIP](https://arxiv.org/pdf/2403.16848)|[repo](<https://github.com/MCG-NJU/MOTIP>)|ECCV 2024 (×)||
|[MLT-Track](https://arxiv.org/pdf/2404.12031)|will be available: [repo](<https://github.com/mzl163/MLS-Track>)|ECCV 2024 (×)|Refer-UE-City 数据集、RMOT|
|[SMOT、BenSMOT、SMOTer](https://arxiv.org/abs/2403.05021)|[repo](<https://github.com/HengLan/SMOT>)|ECCV 2024|语义多目标跟踪 (SMOT) 任务、BenSMOT 数据集、SMOTer 框架|
|[LG-MOT](https://arxiv.org/pdf/2406.04844)|[repo](<https://github.com/WesLee88524/LG-MOT>)|ECCV 2024 (×)|多模态|
|[SPAM](https://arxiv.org/pdf/2404.11426)|无|ECCV 2024|MOT标注生成|
|[BUSCA](https://arxiv.org/pdf/2407.10151)|[repo](<https://github.com/lorenzovaquero/BUSCA>)|ECCV 2024||
|[VETRA](https://elib.dlr.de/205389/1/Hellekes_et_al_2024_VETRA_dataset_preprint.pdf)|[dataset](<https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/vetra>)|ECCV 2024|航空图像、车辆跟踪|
|[Walker](https://eccv.ecva.net/virtual/2024/poster/385)|paper 和 repo 都未发布|ECCV 2024|自监督|
|[JDT3D](https://arxiv.org/pdf/2407.04926)|[repo](<https://github.com/TRAILab/JDT3D>)|ECCV 2024|3D、Track-by-attention (TBA)|
|[OneTrack](https://eccv.ecva.net/virtual/2024/poster/2192)|paper 和 repo 都未发布|ECCV 2024|3D、end-to-end|

## 单目标

| paper | repo | year| jyliu 观点 | kgmao 观点|
| ----- | ---- | --- | ---------- | --------- |
|[PVT++](https://arxiv.org/pdf/2211.11629)|<https://github.com/jaraxxus-me/pvt_pp>|ICCV 2023||
|[Cross-modal Orthogonal High-rank Augmentation for RGB-Event Transformer-trackers](https://arxiv.org/pdf/2307.04129)|<https://github.com/ZHU-Zhiyu/High-Rank_RGB-Event_Tracker>|ICCV 2023|跨模态|
|[MixCycle](https://arxiv.org/pdf/2303.09219)|<https://github.com/Mumuqiao/MixCycle>|ICCV 2023|3D、半监督|
|[F-BDMTrack](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_Foreground-Background_Distribution_Modeling_Transformer_for_Visual_Object_Tracking_ICCV_2023_paper.pdf)|无|ICCV 2023||
|[ROMTrack](https://arxiv.org/pdf/2308.05140)|<https://github.com/dawnyc/ROMTrack>|ICCV 2023||
|[MoMA-M3T](https://arxiv.org/pdf/2308.11607)|<https://github.com/kuanchihhuang/MoMA-M3T>|ICCV 2023|3D|
|[MITS](https://arxiv.org/pdf/2308.13266)|<https://github.com/yoxu515/MITS>|ICCV 2023|Segment+Track|
|[MBP-Track](https://arxiv.org/pdf/2303.05071)|<https://github.com/slothfulxtx/MBPTrack3D>|ICCV 2023|3D|
|[Aba-ViTrack](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Adaptive_and_Background-Aware_Vision_Transformer_for_Real-Time_UAV_Tracking_ICCV_2023_paper.pdf)|<https://github.com/xyyang317/Aba-ViTrack>|ICCV 2023||
|[HiT](https://arxiv.org/pdf/2308.06904)|<https://github.com/kangben258/HiT>|ICCV 2023||
|[SyncTrack](https://arxiv.org/abs/2308.12549)|无|ICCV 2023|3D|
|[CiteTracker](https://arxiv.org/pdf/2308.11322)|<https://github.com/NorahGreen/CiteTracker>|ICCV 2023|多模态|
|[DecoupleTNL](https://openaccess.thecvf.com/content/ICCV2023/papers/Ma_Tracking_by_Natural_Language_Specification_with_Long_Short-term_Context_Decoupling_ICCV_2023_paper.pdf)|无|ICCV 2023|多模态|
|[ZoomTrack](https://openreview.net/pdf?id=DQgTewaKzt)|<https://github.com/Kou-99/ZoomTrack>|NIPS 2023||
|[MixFormerv2](https://openreview.net/pdf?id=8WvYAycmDJ)|<https://github.com/MCG-NJU/MixFormerV2>|NIPS 2023||
|[RFGM-B256](https://openreview.net/pdf?id=On0IDMYKw2)|无|NIPS 2023||
|[BadTrack](https://openreview.net/pdf?id=W9pJx9sFCh)|无|NIPS 2023||
|[BAT](https://arxiv.org/pdf/2312.10611)|<https://github.com/SparkTempest/BAT>|AAAI 2024||
|[M3SOT](https://arxiv.org/pdf/2312.06117)|<https://github.com/ywuchina/TeamCode>|AAAI 2024|3D|
|[StreamTrack](https://ojs.aaai.org/index.php/AAAI/article/view/28196)|无|AAAI 2024|3D|
|[UVLTrack](https://ojs.aaai.org/index.php/AAAI/article/view/28205)|<https://github.com/OpenSpaceAI/UVLTrack>|AAAI 2024|多模态|
|[EVPTrack](https://ojs.aaai.org/index.php/AAAI/article/view/28286)|<https://github.com/GXNU-ZhongLab/EVPTrack>|AAAI 2024||
|[GMMT](https://ojs.aaai.org/index.php/AAAI/article/view/28325)|<https://github.com/Zhangyong-Tang/GMMT>|AAAI 2024|多模态|
|[TATrack](https://ojs.aaai.org/index.php/AAAI/article/view/28352)|TATrack|AAAI 2024|多模态|
|[SCVTrack](https://ojs.aaai.org/index.php/AAAI/article/view/28544)|<https://github.com/zjwhit/SCVTrack>|AAAI 2024|3D|
|[ODTrack](https://ojs.aaai.org/index.php/AAAI/article/view/28591)|<https://github.com/GXNU-ZhongLab/ODTrack>|AAAI 2024||
|[ARTrackV2](https://arxiv.org/pdf/2312.17133)|<https://artrackv2.github.io/>|CVPR 2024||
|[RTracker](https://arxiv.org/pdf/2403.19242v1)|not release|CVPR 2024||
|[QueryNLT](https://arxiv.org/pdf/2403.19975v1)|not release|CVPR 2024|多模态|
|[SpatialTracker](https://arxiv.org/pdf/2404.04319v1)|<https://henry123-boy.github.io/SpaTracker/>|CVPR 2024||
|[SoCL](https://arxiv.org/pdf/2404.09504v1)|无|CVPR 2024||
|[DiffusionTrack](https://openaccess.thecvf.com/content/CVPR2024/papers/Xie_DiffusionTrack_Point_Set_Diffusion_Model_for_Visual_Object_Tracking_CVPR_2024_paper.pdf)|<https://github.com/phiphiphi31/DiffusionTrack>|CVPR 2024||
|[OneTracker](https://arxiv.org/pdf/2403.09634v1)|无|CVPR 2024|多模态、Foundation Models|
|[HIPTrack](https://arxiv.org/pdf/2311.02072)|<https://github.com/WenRuiCai/HIPTrack>|CVPR 2024||
|[UnTrack](https://arxiv.org/pdf/2311.15851)|not release|CVPR 2024|任意模态|
|[SDSTrack](https://arxiv.org/abs/2403.16002v2)|<https://github.com/hoqolo/SDSTrack>|CVPR 2024|多模态|
|[LRR](https://openreview.net/attachment?id=3qo1pJHabg&name=pdf)|<https://github.com/tsingqguo/robustOT>|ICLR 2024|多模态|
|[VastTrack](https://arxiv.org/pdf/2403.03493)|<https://github.com/HengLan/VastTrack>|ECCV 2024 (×)|VastTrack 多模态数据集|
|[DiffTracker](https://arxiv.org/pdf/2407.08394)|无|ECCV 2024|无监督|
|[LoRAT](https://arxiv.org/pdf/2403.05231)|<https://github.com/LitingLin/LoRAT>|ECCV 2024|LoRA|
|SemTrack|<https://sutdcv.github.io/SemTrack/>|ECCV 2024|语义跟踪|
|[HVTrack](https://arxiv.org/pdf/2408.02049)|<https://github.com/Mumuqiao/HVTrack>|ECCV 2024|3D|
|[Diff-Tracker](https://arxiv.org/pdf/2407.08394)|无|ECCV 2024|diffusion、无监督、多模态|
|[FERMT](https://eccv.ecva.net/virtual/2024/poster/1619)|paper 和 repo 都未发布|ECCV 2024|

## 其他
### Tracking
| paper | repo | year| jyliu 观点 | kgmao 观点|
| ----- | ---- | --- | ---------- | --------- |
|[GarmentTracking](https://arxiv.org/pdf/2303.13913)|<https://garment-tracking.robotflow.ai/>|CVPR 2023|服装跟踪|
|[ContourTracking](https://arxiv.org/pdf/2303.08364)|<https://github.com/JunbongJang/contour-tracking>|CVPR 2023|细胞轮廓跟踪|
|[NLOS-Track](https://arxiv.org/pdf/2303.11791)|<https://againstentropy.github.io/NLOS-Track/>|CVPR 2023|非视距 (NLOR) 跟踪|
|[PlanerTrack](https://arxiv.org/pdf/2303.07625)|<https://hengfan2010.github.io/projects/PlanarTrack/>|ICCV 2023|单平面跟踪数据集|
|[MPOT、PRTrack](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Multiple_Planar_Object_Tracking_ICCV_2023_paper.pdf)|<https://zzcheng.top/MPOT/>|ICCV 2023|多平面追踪 (MPOT) 任务、PRTrack框架|
|[Tracking by 3D Model Estimation of Unknown Objects in Videos](https://arxiv.org/pdf/2304.06419)|无|ICCV 2023|
|[TAPIR](https://arxiv.org/pdf/2306.08637)|<https://hengfan2010.github.io/projects/PlanarTrack/>|ICCV 2023|点跟踪|
|[OmniMotion](https://arxiv.org/pdf/2306.05422)|<https://omnimotion.github.io/>|ICCV 2023|点跟踪、伪深度|
|[Context-PIP](https://openreview.net/pdf?id=cnpkzQZaLU)|<https://wkbian.github.io/Projects/Context-PIPs/>|NIPS 2023|点跟踪|
|[KPA-Tracker](https://ojs.aaai.org/index.php/AAAI/article/view/28158/28317)|<https://ojs.aaai.org/index.php/AAAI/article/view/28158/28317>|AAAI 2024|姿态追踪|
|[CodedEvents](https://openaccess.thecvf.com/content/CVPR2024/papers/Shah_CodedEvents_Optimal_Point-Spread-Function_Engineering_for_3D-Tracking_with_Event_Cameras_CVPR_2024_paper.pdf)|无|CVPR 2024||
|[LEAP-VO](https://arxiv.org/pdf/2401.01887)|无|CVPR 2024|点跟踪|
|[DPHMs](https://arxiv.org/pdf/2312.01068)|<https://tangjiapeng.github.io/projects/DPHMs/>|CVPR 2024|人头模型|
|[DecoMotion](https://arxiv.org/pdf/2407.06531)|not release|ECCV 2024|点跟踪|
|[MapTracker](https://arxiv.org/pdf/2403.15951)|<https://map-tracker.github.io/>|ECCV 2024|高清地图 (HD-map)|
|[PapMOT](https://eccv.ecva.net/virtual/2024/poster/1816)|paper 和 repo 都未发布|ECCV 2024|对抗补丁攻击 (Adversarial Patch Attack)、MOT|
|[AADN](https://arxiv.org/pdf/2402.17976)|无|ECCV 2024|对抗防御 (Adversarial Defense)、SOT|
|[DINO-Tracker](https://arxiv.org/pdf/2403.14548)|<https://dino-tracker.github.io/>|ECCV 2024|点跟踪|
|[ultrack](https://arxiv.org/pdf/2308.04526)|<https://github.com/royerlab/ultrack>|ECCV 2024|3D、细胞跟踪|
|[LocoTrack](https://arxiv.org/pdf/2407.15420)|无|ECCV 2024|点跟踪|
|[TAPTR](https://arxiv.org/pdf/2403.13042)|<https://github.com/IDEA-Research/TAPTR>|ECCV 2024|点跟踪|
|[Trackastra](https://arxiv.org/pdf/2405.15700)|<https://github.com/weigertlab/trackastra>|ECCV 2024|细胞跟踪|
|[GMRW](https://arxiv.org/pdf/2409.16288v1)|<https://www.ayshrv.com/gmrw>|ECCV 2024|点跟踪|
|[FastOmniTrack](https://arxiv.org/pdf/2403.17931)|<https://timsong412.github.io/FastOmniTrack/>|ECCV 2024|点跟踪|
|[DecoMotion](https://arxiv.org/pdf/2407.06531)|not release|ECCV 2024|点跟踪|
|[co-tracker](https://arxiv.org/pdf/2307.07635)|<https://arxiv.org/pdf/2307.07635>|ECCV 2024|点跟踪|
|[LaMOT](https://arxiv.org/pdf/2406.08324)|[repo](https://arxiv.org/pdf/2406.08324)|arxiv 2024.6|OVMOT+RMOT|
|[TAO-Amodal](https://arxiv.org/pdf/2312.12433)|[repo](https://tao-amodal.github.io/index.html)|arxiv 2023.12|非模态感知跟踪||

### GNN
| paper | repo | year| jyliu 观点 | kgmao 观点|
| ----- | ---- | --- | ---------- | --------- |
|[Superglue](https://arxiv.org/pdf/1911.11763)|[repo](https://github.com/magicleap/SuperGluePretrainedNetwork)|CVPR 2020 oral|||
