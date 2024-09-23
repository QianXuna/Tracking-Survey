"# Tracking-Survey" 

# 论文列表
|列表名|repo|year（知识库为更新时间）|观点|
|-----|----|-----------------------|-----|
|知识库|[awesome-multiple-object-tracking](https://github.com/luanshiyinyang/awesome-multiple-object-tracking)|2024|常规多目标跟踪|
|知识库（https://arxiv.org/abs/2405.14200）|[Awesome-Multimodal-Object-Tracking](https://github.com/983632847/Awesome-Multimodal-Object-Tracking)|2024|多模态多目标跟踪（MMOT），包括RGBD、RGBT、RGBE、RGBL、混合模态tracking|
|知识库|[awesome-3d-multi-object-tracking-autonomous-driving](https://github.com/MagicTZ/awesome-3d-multi-object-tracking-autonomous-driving)|2022|自动驾驶3D多目标跟踪|
|知识库|[awesome-3d-multi-object-tracking](https://github.com/Po-Jen/awesome-3d-multi-object-tracking)|2022|3D多目标跟踪|
|[ICCV 2023](https://openaccess.thecvf.com/ICCV2023?day=all)|https://blog.csdn.net/CV_Autobot/article/details/132222653 https://github.com/amusi/ICCV2023-Papers-with-Code?tab=readme-ov-file#VThttps://github.com/amusi/ICCV2023-Papers-with-Code?tab=readme-ov-file#VT|2023||
|[CVPR 2023](https://openaccess.thecvf.com/CVPR2023?day=all)|https://zhuanlan.zhihu.com/p/615368672|2023|
|CVPR 2024|https://github.com/52CV/CVPR-2024-Papers?tab=readme-ov-file#25|2024||
|AAAI 2024|https://github.com/DmitryRyumin/AAAI-2024-Papers?tab=readme-ov-file|2024||


# Paper
## 多目标

| paper | repo | year| jyliu 观点 | kgmao 观点|
| ----- | ---- | --- | ---------- | --------- |
|[MotionTrack](https://arxiv.org/pdf/2303.10404)|无|CVPR 2023||
|[GHOST](https://arxiv.org/pdf/2206.04656)|https://github.com/dvl-tum/GHOST|CVPR 2023||
|[OC-SORT](https://arxiv.org/pdf/2203.14360)|https://arxiv.org/pdf/2203.14360|CVPR 2023||
|[FineTrack](https://arxiv.org/pdf/2302.14589)|无|CVPR 2023||
|[RMOT、Refer-KITTI、TransRMOT](https://arxiv.org/pdf/2303.03366)|https://github.com/wudongming97/RMOT|CVPR 2023|指代多目标跟踪 (RMOT) 任务、Refer-KiTTI数据集、TransRMOT框架|
|[MOTRv2](https://arxiv.org/pdf/2211.09791)|https://arxiv.org/pdf/2211.09791|CVPR 2023||
|[ColTrack](https://arxiv.org/pdf/2308.05911)|https://github.com/bytedance/ColTrack|ICCV 2023|低帧率|
|[MeMOTR](https://arxiv.org/pdf/2307.15700)|https://github.com/MCG-NJU/MeMOTR|ICCV 2023||
|[TrackFlow](https://arxiv.org/pdf/2308.11513)|无|ICCV 2023||
|[ReST](https://arxiv.org/pdf/2308.13229)|https://github.com/chengche6230/ReST|ICCV 2023|multi-camera|
|[FUS3D](https://openaccess.thecvf.com/content/ICCV2023/papers/Heitzinger_A_Fast_Unified_System_for_3D_Object_Detection_and_Tracking_ICCV_2023_paper.pdf)|https://github.com/theitzin/FUS3D|ICCV 2023|3D 检测、3D tracking|
|[HD-AMOT](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Heterogeneous_Diversity_Driven_Active_Learning_for_Multi-Object_Tracking_ICCV_2023_paper.pdf)|无|ICCV 2023||
|[3DMOTFormer](https://arxiv.org/pdf/2308.06635)|https://github.com/dsx0511/3DMOTFormer|ICCV 2023|3D|
|[OC-MOT](https://arxiv.org/pdf/2309.00233))|https://github.com/amazon-science/object-centric-multiple-object-tracking|ICCV 2023|无监督|
|[TrajectoryFormer](https://arxiv.org/pdf/2306.05888)|https://github.com/poodarchu/EFG|ICCV 2023|3D|
|[SportsMOT、MixSort](https://arxiv.org/pdf/2304.05170)|https://deeperaction.github.io/datasets/sportsmot.html|ICCV 2023|SportsMOT数据集、MixSort框架|
|[U2MOT](https://arxiv.org/pdf/2307.15409)|https://github.com/alibaba/u2mot/|ICCV 2023|无监督|
|[DQTrack](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_End-to-end_3D_Tracking_with_Decoupled_Queries_ICCV_2023_paper.pdf)|https://sites.google.com/view/dqtrack|ICCV 2023|3D|
|[DARTH](https://openaccess.thecvf.com/content/ICCV2023/papers/Segu_DARTH_Holistic_Test-time_Adaptation_for_Multiple_Object_Tracking_ICCV_2023_paper.pdf)|https://github.com/mattiasegu/darth|ICCV 2023|域偏移|
|[UCSL](https://arxiv.org/pdf/2309.00942)|无|ICCV 2023|无监督|
|[Type-to-Track](https://arxiv.org/pdf/2305.13495)|https://github.com/uark-cviu/Type-to-Track|NIPS 2023||
|[Hybrid-SORT](https://browse.arxiv.org/pdf/2308.00783)|https://github.com/ymzis69/HybridSORT|AAAI 2024||
|[UCMCTrack](https://browse.arxiv.org/pdf/2312.08952)|https://github.com/corfyi/UCMCTrack|AAAI 2024||
|[SRTrack](https://ojs.aaai.org/index.php/AAAI/article/view/28115)|https://github.com/lzzppp/SR-Track|AAAI 2024||
|[Multi-Scene Generalized Trajectory Global Graph Solver with Composite Nodes for Multiple Object Tracking](https://arxiv.org/pdf/2312.08951)|无|AAAI 2024||
|[DiffusionTrack](https://arxiv.org/pdf/2308.09905)|https://github.com/RainBowLuoCS/DiffusionTrack|AAAI 2024||
|[SlowTrack](https://ojs.aaai.org/index.php/AAAI/article/view/28200)|无|AAAI 2024||
|[SMILEtrack](https://ojs.aaai.org/index.php/AAAI/article/view/28386)|https://github.com/pingyang1117/SMILEtrack_Official|AAAI 2024||
|[NetTrack](https://arxiv.org/pdf/2403.11186v1)|https://george-zhuang.github.io/nettrack/|CVPR 2024||
|[LMOT](https://arxiv.org/pdf/2405.06600)|not release|CVPR 2024||
|[GeneralTrack](https://arxiv.org/pdf/2406.00429)|https://github.com/qinzheng2000/GeneralTrack|CVPR 2024||
|[ADATrack](https://arxiv.org/pdf/2405.08909)|https://github.com/dsx0511/ADA-Track|CVPR 2024|3D、multi-camera|
|[DeconfuseTrack](https://arxiv.org/pdf/2403.02767)|无|CVPR 2024||
|[Delving into the Trajectory Long-tail Distribution for Muti-object Tracking](https://arxiv.org/abs/2403.04700v2)|https://github.com/chen-si-jia/Trajectory-Long-tail-Distribution-for-MOT|CVPR 2024||
|[Self-Supervised Multi-Object Tracking with Path Consistency](https://arxiv.org/pdf/2404.05136v1)|https://github.com/amazon-science/path-consistency|CVPR 2024||
|[DiffMOT](https://arxiv.org/abs/2403.02075)|https://diffmot.github.io/|CVPR 2024||
|[iKUN](https://arxiv.org/pdf/2312.16245)|https://github.com/dyhBUPT/iKUN|CVPR 2024||
|[SLack](https://arxiv.org/pdf/2409.11235)|https://github.com/siyuanliii/SLAck|ECCV 2024|开放词汇|


## 单目标

| paper | repo | year| jyliu 观点 | kgmao 观点|
| ----- | ---- | --- | ---------- | --------- |
|[PVT++](https://arxiv.org/pdf/2211.11629)|https://github.com/jaraxxus-me/pvt_pp|ICCV 2023||
|[Cross-modal Orthogonal High-rank Augmentation for RGB-Event Transformer-trackers](https://arxiv.org/pdf/2307.04129)|https://github.com/ZHU-Zhiyu/High-Rank_RGB-Event_Tracker|ICCV 2023|跨模态|
|[MixCycle](https://arxiv.org/pdf/2303.09219)|https://github.com/Mumuqiao/MixCycle|ICCV 2023|3D、半监督|
|[F-BDMTrack](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_Foreground-Background_Distribution_Modeling_Transformer_for_Visual_Object_Tracking_ICCV_2023_paper.pdf)|无|ICCV 2023||
|[ROMTrack](https://arxiv.org/pdf/2308.05140)|https://github.com/dawnyc/ROMTrack|ICCV 2023||
|[MoMA-M3T](https://arxiv.org/pdf/2308.11607)|https://github.com/kuanchihhuang/MoMA-M3T|ICCV 2023|3D|
|[MITS](https://arxiv.org/pdf/2308.13266)|https://github.com/yoxu515/MITS|ICCV 2023|Segment+Track|
|[MBP-Track](https://arxiv.org/pdf/2303.05071)|https://github.com/slothfulxtx/MBPTrack3D|ICCV 2023|3D|
|[OmniMotion](https://arxiv.org/pdf/2306.05422)|https://omnimotion.github.io/|ICCV 2023|伪深度|
|[Aba-ViTrack](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Adaptive_and_Background-Aware_Vision_Transformer_for_Real-Time_UAV_Tracking_ICCV_2023_paper.pdf)|https://github.com/xyyang317/Aba-ViTrack|ICCV 2023||
|[HiT](https://arxiv.org/pdf/2308.06904)|https://github.com/kangben258/HiT|ICCV 2023||
|[SyncTrack](https://arxiv.org/abs/2308.12549)|无|ICCV 2023|3D|
|[CiteTracker](https://arxiv.org/pdf/2308.11322)|https://github.com/NorahGreen/CiteTracker|ICCV 2023|多模态|
|[DecoupleTNL](https://openaccess.thecvf.com/content/ICCV2023/papers/Ma_Tracking_by_Natural_Language_Specification_with_Long_Short-term_Context_Decoupling_ICCV_2023_paper.pdf)|无|ICCV 2023|多模态|
|[BAT](https://arxiv.org/pdf/2312.10611)|https://github.com/SparkTempest/BAT|AAAI 2024||
|[M3SOT](https://arxiv.org/pdf/2312.06117)|https://github.com/ywuchina/TeamCode|AAAI 2024|3D|
|[StreamTrack](https://ojs.aaai.org/index.php/AAAI/article/view/28196)|无|AAAI 2024|3D|
|[UVLTrack](https://ojs.aaai.org/index.php/AAAI/article/view/28205)|https://github.com/OpenSpaceAI/UVLTrack|AAAI 2024|多模态|
|[EVPTrack](https://ojs.aaai.org/index.php/AAAI/article/view/28286)|https://github.com/GXNU-ZhongLab/EVPTrack|AAAI 2024||
|[GMMT](https://ojs.aaai.org/index.php/AAAI/article/view/28325)|https://github.com/Zhangyong-Tang/GMMT|AAAI 2024|多模态|
|[TATrack](https://ojs.aaai.org/index.php/AAAI/article/view/28352)|TATrack|AAAI 2024|多模态|
|[SCVTrack](https://ojs.aaai.org/index.php/AAAI/article/view/28544)|https://github.com/zjwhit/SCVTrack|AAAI 2024|3D|
|[ODTrack](https://ojs.aaai.org/index.php/AAAI/article/view/28591)|https://github.com/GXNU-ZhongLab/ODTrack|AAAI 2024||
|[ARTrackV2](https://arxiv.org/pdf/2312.17133)|https://artrackv2.github.io/|CVPR 2024||
|[RTracker](https://arxiv.org/pdf/2403.19242v1)|not release|CVPR 2024||
|[QueryNLT](https://arxiv.org/pdf/2403.19975v1)|not release|CVPR 2024|多模态|
|[SpatialTracker](https://arxiv.org/pdf/2404.04319v1)|https://henry123-boy.github.io/SpaTracker/|CVPR 2024||
|[SoCL](https://arxiv.org/pdf/2404.09504v1)|无|CVPR 2024||
|[DiffusionTrack](https://openaccess.thecvf.com/content/CVPR2024/papers/Xie_DiffusionTrack_Point_Set_Diffusion_Model_for_Visual_Object_Tracking_CVPR_2024_paper.pdf)|https://github.com/phiphiphi31/DiffusionTrack|CVPR 2024||
|[OneTracker](https://arxiv.org/pdf/2403.09634v1)|无|CVPR 2024|多模态、Foundation Models|
|[HIPTrack](https://arxiv.org/pdf/2311.02072)|https://github.com/WenRuiCai/HIPTrack|CVPR 2024||
|[UnTrack](https://arxiv.org/pdf/2311.15851)|not release|CVPR 2024|任意模态|
|[SDSTrack](https://arxiv.org/abs/2403.16002v2)|https://github.com/hoqolo/SDSTrack|CVPR 2024|多模态|
|[LRR](https://openreview.net/attachment?id=3qo1pJHabg&name=pdf)|https://github.com/tsingqguo/robustOT|ICLR 2024|多模态|


## 其他
| paper | repo | year| jyliu 观点 | kgmao 观点|
| ----- | ---- | --- | ---------- | --------- |
|[PlanerTrack](https://arxiv.org/pdf/2303.07625)|https://hengfan2010.github.io/projects/PlanarTrack/|ICCV 2023|单平面跟踪数据集|
|[MPOT、PRTrack](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Multiple_Planar_Object_Tracking_ICCV_2023_paper.pdf)|https://zzcheng.top/MPOT/|ICCV 2023|多平面追踪 (MPOT) 任务、PRTrack框架|
|[Tracking by 3D Model Estimation of Unknown Objects in Videos](https://arxiv.org/pdf/2304.06419)|无|ICCV 2023|
|[TAPIR](https://arxiv.org/pdf/2306.08637)|https://hengfan2010.github.io/projects/PlanarTrack/|ICCV 2023|点跟踪|
|[KPA-Tracker](https://ojs.aaai.org/index.php/AAAI/article/view/28158/28317)|https://ojs.aaai.org/index.php/AAAI/article/view/28158/28317|AAAI 2024|姿态追踪|
|[CodedEvents](https://openaccess.thecvf.com/content/CVPR2024/papers/Shah_CodedEvents_Optimal_Point-Spread-Function_Engineering_for_3D-Tracking_with_Event_Cameras_CVPR_2024_paper.pdf)|无|CVPR 2024||
|[LEAP-VO](https://arxiv.org/pdf/2401.01887)|无|CVPR 2024|点跟踪|
|[DPHMs](https://arxiv.org/pdf/2312.01068)|https://tangjiapeng.github.io/projects/DPHMs/|CVPR 2024|人头模型|