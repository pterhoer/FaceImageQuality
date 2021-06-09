# Face Image Quality Assessment

***15.05.2020*** _SER-FIQ (CVPR2020) was added._

***18.05.2020*** _Bias in FIQ (IJCB2020) was added._



## SER-FIQ: Unsupervised Estimation of Face Image Quality Based on Stochastic Embedding Robustness



IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2020

* [Research Paper](https://arxiv.org/abs/2003.09373)
* [Implementation on ArcFace](face_image_quality.py)
* [Video](https://www.youtube.com/watch?v=soW_Gg4NElc)


## Table of Contents 

- [Abstract](#abstract)
- [Key Points](#key-points)
- [Results](#results)
- [Installation](#installation)
- [Bias in Face Quality Assessment](#bias-in-face-quality-assessment)
- [Citing](#citing)
- [Acknowledgement](#acknowledgement)
- [License](#license)

## Abstract

<img src="CVPR_2020_teaser_1200x1200.gif" width="400" height="400" align="right">

Face image quality is an important factor to enable high-performance face recognition systems. Face quality assessment aims at estimating the suitability of a face image for recognition. Previous works proposed supervised solutions that require artificially or human labelled quality values. However, both labelling mechanisms are error-prone as they do not rely on a clear definition of quality and may not know the best characteristics for the utilized face recognition system. Avoiding the use of inaccurate quality labels, we proposed a novel concept to measure face quality based on an arbitrary face recognition model. By determining the embedding variations generated from random subnetworks of a face model, the robustness of a sample representation and thus, its quality is estimated. The experiments are conducted in a cross-database evaluation setting on three publicly available databases. We compare our proposed solution on two face embeddings against six state-of-the-art approaches from academia and industry. The results show that our unsupervised solution outperforms all other approaches in the majority of the investigated scenarios. In contrast to previous works, the proposed solution shows a stable performance over all scenarios. Utilizing the deployed face recognition model for our face quality assessment methodology avoids the training phase completely and further outperforms all baseline approaches by a large margin. Our solution can be easily integrated into current face recognition systems and can be modified to other tasks beyond face recognition.

## Key Points

- Quality assessment with SER-FIQ is most effective when the quality measure is based on the deployed face recognition network, meaning that **the quality estimation and the recognition should be performed on the same network**. This way the quality estimation captures the same decision patterns as the face recognition system.
- To get accurate quality estimations, the underlying face recognition network for SER-FIQ should be **trained with dropout**. This is suggested since our solution utilizes the robustness against dropout variations as a quality indicator.
- The provided code is only a demonstration on how SER-FIQ can be utilized. The main contribution of SER-FIQ is the novel concept of measuring face image quality.
- If the last layer contains dropout, it is sufficient to repeat the stochastic forward passes only on this layer. This significantly reduces the computation time to a time span of a face template generation.

## Results

Face image quality assessment results are shown below on LFW (left) and Adience (right). SER-FIQ (same model) is based on ArcFace and shown in red. The plots show the FNMR at ![\Large 10^{-3}](https://latex.codecogs.com/gif.latex?\inline&space;10^{-3}) FMR as recommended by the [best practice guidelines](https://op.europa.eu/en/publication-detail/-/publication/e81d082d-20a8-11e6-86d0-01aa75ed71a1) of the European Border Guard Agency Frontex. For more details and results, please take a look at the paper.

<img src="FQA-Results/001FMR_lfw_arcface.png" width="430" >  <img src="FQA-Results/001FMR_adience_arcface.png" width="430" >

## Installation

We recommend Anaconda to install the required packages.
This can be done by creating an virtual environment via

```shell
conda env create -f environment.yml
```

or by manually installing the following packages.


```shell
conda create -n serfiq python=3.6.9
conda install cudatoolkit
conda install cudnn
conda install tensorflow=1.14.0
conda install mxnet
conda install mxnet-gpu
conda install tqdm
conda install -c conda-forge opencv
conda install -c anaconda scikit-learn
conda install -c conda-forge scikit-image
conda install keras=2.2.4
```

After the required packages have been installed, also download the [Insightface codebase at the needed git point in the repository history](https://github.com/deepinsight/insightface/tree/60bb5829b1d76bfcec7930ce61c41dde26413279) to a location of your choice and extract the archive if necessary.

We will refer to this location as _$Insightface_ in the following. 

The path to the Insightface repository must be passed to the [InsightFace class in face_image_quality.py](https://github.com/pterhoer/FaceImageQuality/blob/b59b2ec3c58429ee867dee25a4d8165b9c65d304/face_image_quality.py#L25). To avoid any problems, absolute paths can be used. Our InsightFace class automatically imports the required dependencies from the Insightface repository.
```
insightface = InsightFace(insightface_path = $Insightface) # Repository-path as parameter
```
[Please be aware to change the location in our example code according to your setup](https://github.com/pterhoer/FaceImageQuality/blob/b59b2ec3c58429ee867dee25a4d8165b9c65d304/serfiq_example.py#L9).

A pre-trained Arcface model is also required. We recommend using the "_LResNet100E-IR,ArcFace@ms1m-refine-v2_" model. [This can be downloaded from the Insightface Model-Zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo#31-lresnet100e-irarcfacems1m-refine-v2).

Extract the downloaded _model-0000.params_ and _model-symbol.json_ to the following location on your computer:
```
$Insightface/models/
```

After following these steps you can activate your environment (default: _conda activate serfiq_) and run the [example code](serfiq_example.py).

The implementation for SER-FIQ based on ArcFace can be found here: [Implementation](face_image_quality.py). <br/>
In the [Paper](https://arxiv.org/abs/2003.09373), this is refered to _SER-FIQ (same model) based on ArcFace_. <br/>



## Bias in Face Quality Assessment

The best face quality assessment performance is achieved when the quality assessment solutions build on the templates of the deployed face recognition system.
In our work on ([Face Quality Estimation and Its Correlation to Demographic and Non-Demographic Bias in Face Recognition](https://arxiv.org/abs/2004.01019)), we showed that this lead to a bias transfer from the face recognition system to the quality assessment solution.
On all investigated quality assessment approaches, we observed performance differences based on on demographics and non-demographics of the face images.


<img src="/Bias-FQA/stack_SER-FIQ_colorferet_arcface_pose.png" width="270"> <img src="/Bias-FQA/stack_SER-FIQ_colorferet_arcface_ethnic.png" width="270"> <img src="/Bias-FQA/stack_SER-FIQ_adience_arcface_age.png" width="270">

<img src="/Bias-FQA/quality_distribution_SER-FIQ_colorferet_arcface_pose.png" width="270"> <img src="/Bias-FQA/quality_distribution_SER-FIQ_colorferet_arcface_ethnic.png" width="270"> <img src="/Bias-FQA/quality_distribution_SER-FIQ_adience_arcface_age.png" width="270">



## Citing

If you use this code, please cite the following papers.


```
@inproceedings{DBLP:conf/cvpr/TerhorstKDKK20,
  author    = {Philipp Terh{\"{o}}rst and
               Jan Niklas Kolf and
               Naser Damer and
               Florian Kirchbuchner and
               Arjan Kuijper},
  title     = {{SER-FIQ:} Unsupervised Estimation of Face Image Quality Based on
               Stochastic Embedding Robustness},
  booktitle = {2020 {IEEE/CVF} Conference on Computer Vision and Pattern Recognition,
               {CVPR} 2020, Seattle, WA, USA, June 13-19, 2020},
  pages     = {5650--5659},
  publisher = {{IEEE}},
  year      = {2020},
  url       = {https://doi.org/10.1109/CVPR42600.2020.00569},
  doi       = {10.1109/CVPR42600.2020.00569},
  timestamp = {Tue, 11 Aug 2020 16:59:49 +0200},
  biburl    = {https://dblp.org/rec/conf/cvpr/TerhorstKDKK20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

```
@inproceedings{DBLP:conf/icb/TerhorstKDKK20,
  author    = {Philipp Terh{\"{o}}rst and
               Jan Niklas Kolf and
               Naser Damer and
               Florian Kirchbuchner and
               Arjan Kuijper},
  title     = {Face Quality Estimation and Its Correlation to Demographic and Non-Demographic
               Bias in Face Recognition},
  booktitle = {2020 {IEEE} International Joint Conference on Biometrics, {IJCB} 2020,
               Houston, TX, USA, September 28 - October 1, 2020},
  pages     = {1--11},
  publisher = {{IEEE}},
  year      = {2020},
  url       = {https://doi.org/10.1109/IJCB48548.2020.9304865},
  doi       = {10.1109/IJCB48548.2020.9304865},
  timestamp = {Thu, 14 Jan 2021 15:14:18 +0100},
  biburl    = {https://dblp.org/rec/conf/icb/TerhorstKDKK20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}


```

If you make use of our SER-FIQ implementation based on ArcFace, please additionally cite the original ![ArcFace module](https://github.com/deepinsight/insightface).

## Acknowledgement

This research work has been funded by the German Federal Ministry of Education and Research and the Hessen State Ministry for Higher Education, Research and the Arts within their joint support of the National Research Center for Applied Cybersecurity ATHENE. 

## License 

This project is licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.
Copyright (c) 2020 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt
