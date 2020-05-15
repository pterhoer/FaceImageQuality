# Face Image Quality Assessment

***15.05.2020*** _SER-FIQ (CVPR2020) was added._

## SER-FIQ: Unsupervised Estimation of Face Image Quality Based on Stochastic Embedding Robustness


IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2020
* [Research Paper](https://arxiv.org/abs/2003.09373)
* [Implementation on ArcFace](face_image_quality.py)

***Abstract***

<img src="CVPR_2020_teaser_1200x1200.gif" width="400" height="400" align="right">

Face image quality is an important factor to enable high-performance face recognition systems. Face quality assessment aims at estimating the suitability of a face image for recognition. Previous works proposed supervised solutions that require artificially or human labelled quality values. However, both labelling mechanisms are error-prone as they do not rely on a clear definition of quality and may not know the best characteristics for the utilized face recognition system. Avoiding the use of inaccurate quality labels, we proposed a novel concept to measure face quality based on an arbitrary face recognition model. By determining the embedding variations generated from random subnetworks of a face model, the robustness of a sample representation and thus, its quality is estimated. The experiments are conducted in a cross-database evaluation setting on three publicly available databases. We compare our proposed solution on two face embeddings against six state-of-the-art approaches from academia and industry. The results show that our unsupervised solution outperforms all other approaches in the majority of the investigated scenarios. In contrast to previous works, the proposed solution shows a stable performance over all scenarios. Utilizing the deployed face recognition model for our face quality assessment methodology avoids the training phase completely and further outperforms all baseline approaches by a large margin. Our solution can be easily integrated into current face recognition systems and can be modified to other tasks beyond face recognition.

***Key points***

<img src="3065-1min.mp4" width="400" align="right">

- Quality assessment with SER-FIQ is most effective when the quality measure is based on the deployed face recognition network, meaning that **the quality estimation and the recognition should be performed on the same network**. This way the quality estimation captures the same decision patterns than the face recognition system.
- To get accurate quality estimations, the underlying face recognition network for SER-FIQ should be **trained with dropout**. This is suggested since our solution utilizes the robustness against dropout variations as a quality indicator.
- The provided code is only a demonstration how SER-FIQ can be utilized. The contribution of SER-FIQ is the novel concept of measuring face quality.
- If the last layer contains dropout, it is sufficient to repeat the stochastic forward passes only on this layer. This significantly reduces the computation time to a time span of a face template generation.

***Bias in Face Quality Assessment***

The best face quality assessment performance is achieved when the quality assessment solutions build on the templates of the deployed face recognition system.
In our work on *Face Quality Estimation and Its Correlation to Demographic and Non-Demographic Bias in Face Recognition* * [Paper](https://arxiv.org/abs/2004.01019), we showed that this lead to a bias transfoer from the face recognition system to the quality assessment.
On all investigated quality assessment approaches, we observed performance differences based on on demographics and non-demographics of the face images.


<img src="/Bias-FQA/stack_SER-FIQ_colorferet_arcface_pose.png" width="250"> <img src="/Bias-FQA/stack_SER-FIQ_colorferet_arcface_ethnic.png" width="250"> <img src="/Bias-FQA/stack_SER-FIQ_adience_arcface_age.png" width="250">

<img src="/Bias-FQA/quality_distribution_SER-FIQ_colorferet_arcface_pose.png" width="250"> <img src="/Bias-FQA/quality_distribution_SER-FIQ_colorferet_arcface_ethnic.png" width="250"> <img src="/Bias-FQA/quality_distribution_SER-FIQ_adience_arcface_age.png" width="250">



***Citing***

If you find our work useful, please consider citing the following works.
If you make use of our SER-FIQ implementation based on ArcFace, please additionally cite the original ![ArcFace paper](https://github.com/deepinsight/insightface).

```
@article{DBLP:journals/corr/abs-2003-09373,
  author    = {Philipp Terh{\"{o}}rst and
               Jan Niklas Kolf and
               Naser Damer and
               Florian Kirchbuchner and
               Arjan Kuijper},
  title     = {{SER-FIQ:} Unsupervised Estimation of Face Image Quality Based on
               Stochastic Embedding Robustness},
  journal   = {CoRR},
  volume    = {abs/2003.09373},
  year      = {2020},
  url       = {https://arxiv.org/abs/2003.09373},
  archivePrefix = {arXiv},
  eprint    = {2003.09373},
  timestamp = {Tue, 24 Mar 2020 16:42:29 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2003-09373.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

```
@article{DBLP:journals/corr/abs-2004-01019,
  author    = {Philipp Terh{\"{o}}rst and
               Jan Niklas Kolf and
               Naser Damer and
               Florian Kirchbuchner and
               Arjan Kuijper},
  title     = {Face Quality Estimation and Its Correlation to Demographic and Non-Demographic
               Bias in Face Recognition},
  journal   = {CoRR},
  volume    = {abs/2004.01019},
  year      = {2020},
  url       = {https://arxiv.org/abs/2004.01019},
  archivePrefix = {arXiv},
  eprint    = {2004.01019},
  timestamp = {Wed, 08 Apr 2020 17:08:25 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2004-01019.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Acknowledgement

This research work has been funded by the German Federal Ministry of Education and Research and the Hessen State Ministry for Higher Education, Research and the Arts within their joint support of the National Research Center for Applied Cybersecurity ATHENE. 

## Licence 

This project is licensed under the terms of the ... license.
Copyright (c) 2020 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt
