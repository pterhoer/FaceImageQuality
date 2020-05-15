# Face Image Quality Assessment

***15.05.2020*** _SER-FIQ (CVPR2020) was added._

## SER-FIQ: Unsupervised Estimation of Face Image Quality Based on Stochastic Embedding Robustness
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2020
* [Research Paper](https://arxiv.org/abs/2003.09373)
* [Implementation on ArcFace](face_image_quality.py)

***Abstract***

Face image quality is an important factor to enable high-performance face recognition systems. Face quality assessment aims at estimating the suitability of a face image for recognition. Previous works proposed supervised solutions that require artificially or human labelled quality values. However, both labelling mechanisms are error-prone as they do not rely on a clear definition of quality and may not know the best characteristics for the utilized face recognition system. Avoiding the use of inaccurate quality labels, we proposed a novel concept to measure face quality based on an arbitrary face recognition model. By determining the embedding variations generated from random subnetworks of a face model, the robustness of a sample representation and thus, its quality is estimated. The experiments are conducted in a cross-database evaluation setting on three publicly available databases. We compare our proposed solution on two face embeddings against six state-of-the-art approaches from academia and industry. The results show that our unsupervised solution outperforms all other approaches in the majority of the investigated scenarios. In contrast to previous works, the proposed solution shows a stable performance over all scenarios. Utilizing the deployed face recognition model for our face quality assessment methodology avoids the training phase completely and further outperforms all baseline approaches by a large margin. Our solution can be easily integrated into current face recognition systems and can be modified to other tasks beyond face recognition.

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

