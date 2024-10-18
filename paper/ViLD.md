# OPEN-VOCABULARY OBJECT DETECTION VIA VISION AND LANGUAGE KNOWLEDGE DISTILLATION
https://arxiv.org/pdf/2104.13921  
https://zhuanlan.zhihu.com/p/369464298

<center><img src=../images/image-52.png style="zoom:50%"></center>
<center><img src=../images/image-55.png style="zoom:50%"></center>
上来自知乎  

<center><img src=../images/image-54.png style="zoom:100%"></center>
<center><img src=../images/image-53.png style="zoom:50%"></center>
上来自知乎  

- region embeddings为Faster R-CNN head输出的
- 图(b)中左半部分叫ViLD-text，右半部分叫ViLD-image
  - ViLD-image: region embeddings是蒸馏自CLIP image embeddings，image embeddings是由M个候选框都通过Crop、resize、image encoder获得的，而CLIP也不会知道这个图片是base还是novel类别的，所以image embeddings的蒸馏出的region embeddings是通过base和novel类别训练的
  - ViLD-text: 
    - 训练阶段，Text embeddings是将所有的base类别送入到CLIP text encoder产生的，从而将region embeddings和text embeddings做内积就类似于CLIP输出的相似度，加一个softmax就可以得到最终分类，与监督信号 $y_r$ 做交叉熵形成监督
      - 个人理解：由于ViLD-text和ViLD-image受监督的类别不同，所以候选框个数也不同 (N、M)
    - 推理阶段，Text embeddings是将所有的base+novel类别送入到CLIP text encoder产生的，如图2所示
    - 作者认为模型是从base类别的标注中训练的，而是在base+novel类别上测试的，所以确实是开放词汇检测任务

问题：
- 推理阶段怎么区分背景和novel类别目标
  - 回答：用base类别训练RPN，也能泛化到novel类别上，参考ViLD论文：
    <center><img src=../images/image-153.png style="zoom:50%"></center>
