# Awesome Long-Tailed Learning 

A curated list of awesome deep long-tailed learning resources.

Recently, we released *[Deep Long-Tailed Learning: A Survey]()* to the community. In this survey, we reviewed recent advances of long-tailed learning based on deep neural networks. To be specific, existing deep long-tailed learning studies can be grouped into three main categories (i.e., class re-balancing, information augmentation and module improvement), which can be further classified into nine sub-categories (as shown in below figure). We also empirically analyzeed several state-of-the-art methods by evaluating to what extent they address the issue of class imbalance. We concluded the survey by highlighting important applications of deep long-tailed learning and identifying several promising directions for future research. After completing this survey, we decided to release the collected long-tailed learning resources, hoping to push the development of the community. We will keep updating our survey and this repository. If you have any questions or suggestions, please feel free to contact us.

<p align="center">
<img src="Taxonomy.png" width=800>
</p>

## Type of Long-tailed Learning

| Symbol    | `Sampling`          | `CSL`           | `LA`                   | `TL`                 | `Aug`                  | 
|:----------- |:-------------:|:--------------:|:----------------------: |:---------------------:|:----------------------:| 
| Type | Re-Sampling | Cost-sensitive Learning | Logit Adjustment | Transfer Learning | Data Augmentation | 

| Symbol    | `RL`          | `CD`           | `DT`                   | `Ensemble`                 | `other`                  | 
|:----------- |:-------------:|:--------------:|:----------------------: |:---------------------:|:----------------------:| 
| Type | Representation Learning | Classifier Design | Decoupled Training | Ensemble Learning | Other Types | 

## Papers

### 2021

| Title    | Venue    | Year | Type     | Code     | 
|:-------- |:--------:|:--------:|:--------:|:--------:|
[Video-LT]() | ICCV  | 2021 | `Sampling`     |       | 
[LOCE]() | ICCV  | 2021 | `Sampling`,`CSL`     |       |  
[GIST]() | ICCV  | 2021 | `Sampling`,`TL`, `DC`      |       |  
[FASA]() | ICCV  | 2021 | `Sampling`,`CSL`     |       |  
[ACE]() | ICCV  | 2021 | `Sampling`,`Ensemble`     |       |  
[DARS]() | ICCV  | 2021 | `TL`     |       |  
[SSD]() | ICCV  | 2021 | `TL`     |       |  
[DiVE]() | ICCV  | 2021 | `TL`     |       |  
[MosaicOS]() | ICCV  | 2021 | `TL`     |       |  
[PaCo]() | ICCV  | 2021 | `RL`     |       |  
[DRO-LT]() | ICCV  | 2021 | `RL`     |       | 
[DT2]() | ICCV  | 2021 | `DT`     |       |  
[Delving into deep imbalanced regression](https://arxiv.org/pdf/2102.09554.pdf) | ICML  | 2021 | `Other`     |    [Official](https://github.com/YyzHarry/imbalanced-regression)   |  
[LTML]() | CVPR  | 2021 | `Sampling`,`Ensemble` |       |
[Equalization loss v2]() | CVPR  | 2021 | `CSL`  |       | 
[Seesaw loss]() | CVPR  | 2021 | `CSL`  |       | 
[ACSL]() | CVPR  | 2021 | `CSL`  |       | 
[PML]() | CVPR  | 2021 | `CSL`  |       | 
[LADE]() | CVPR  | 2021 | `CSL`,`LA`  |       | 
[Adversarial robustness under long-tailed distribution](https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_Adversarial_Robustness_Under_Long-Tailed_Distribution_CVPR_2021_paper.pdf) | CVPR  | 2021 | `CSL`,`LA`,`CD`  |    [Official](https://github.com/wutong16/Adversarial_Long-Tail)     | 
[DisAlign]() | CVPR  | 2021 | `CSL`,`LA`,`DT`  |       | 
[MiSLAS]() | CVPR  | 2021 | `CSL`,`Aug`,`DT`  |       | 
[CReST]() | CVPR  | 2021 | `TL`  |       | 
[Conceptual 12M]() | CVPR  | 2021 | `TL`  |       | 
[RSG]() | CVPR  | 2021 | `TL`,`Aug`  |       |
[MetaSAug]() | CVPR  | 2021 | `Aug`  |       |
[Hybrid]() | CVPR  | 2021 | `RL`  |       |  
[Unsupervised discovery]() | CVPR  | 2021 | `RL`  |       |  
[Long-tail learning via logit adjustment](https://openreview.net/pdf?id=37nvvqkCo5) | ICLR  | 2021 | `LA`     | Official    | 
[RIDE]() | ICLR  | 2021 | `TL`,`Ensemble`  |       |  
[KCL]() | ICLR  | 2021 | `RL`,`DT`  |       |  

### 2020

| Title    | Venue    | Year | Type     | Code     | 
|:-------- |:--------:|:--------:|:--------:|:--------:|
[Balanced Meta-Softmax]() | NeurIPS  | 2020 | `Sampling`,`CSL`     |       | 
[UNO-IC]() | NeurIPS  | 2020 | `LA`     |       | 
[De-confound-TDE]() | NeurIPS  | 2020 | `LA`,`CD`     |       | 
[SSP]() | NeurIPS  | 2020 | `TL`,`RA`     |       | 
[SimCal]() | ECCV  | 2020 | `Sampling`,`DT`,`Ensemble`     |       |
[PRS]() | ECCV  | 2020 | `Sampling`      |       |
[Distribution-balanced loss]() | ECCV  | 2020 | `CSL`     |       |
[OFA]() | ECCV  | 2020 | `TL`,`Aug`,`DT`     |       |
[LFME]() | ECCV  | 2020 | `TL`,`Ensemble`     |       |
[Deep-RTC]() | ECCV  | 2020 | `CD`     |       |
[Learning to segment the tail](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hu_Learning_to_Segment_the_Tail_CVPR_2020_paper.pdf) | CVPR  | 2020 | `Sampling`,`TL`     |   [Official](https://github.com/JoyHuYY1412/LST_LVIS)    |
[BBN]() | CVPR  | 2020 | `Sampling`,`Ensemble`     |       |
[BAGS]() | CVPR  | 2020 | `Sampling`,`Ensemble`     |       |
[Domain adaptation]() | CVPR  | 2020 | `CSL`     |       |
[Equalization loss]() | CVPR  | 2020 | `CSL`     |       |
[DBM]() | CVPR  | 2020 | `CSL`     |       |
[M2m]() | CVPR  | 2020 | `TL`,`Aug`     |       |
[LEAP]() | CVPR  | 2020 | `TL`,`Aug`,`RL`     |       |
[IEM]() | CVPR  | 2020 | `RL`     |       |
[Decoupling]() | ICLR  | 2020 | `Sampling`,`CSL`,`RL`,`CD`,`DT`     |       |


### 2019

| Title    | Venue    | Year | Type     | Code     | 
|:-------- |:--------:|:--------:|:--------:|:--------:|
[Meta-weight-net: Learning an explicit mapping for sample weighting](https://proceedings.neurips.cc/paper/2019/file/e58cc5ca94270acaceed13bc82dfedf7-Paper.pdf) | NeurIPS  | 2019 | `CSL`     |   [Official](https://github.com/xjtushujun/meta-weight-net)       | 
[LDAM]() | NeurIPS  | 2019 | `CSL`     |       | 
[Dynamic curriculum learning for imbalanced data classification](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Dynamic_Curriculum_Learning_for_Imbalanced_Data_Classification_ICCV_2019_paper.pdf) | ICCV  | 2019 | `Sampling`     |       |
[Class-balanced loss based on effective number of samples](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf) | CVPR  | 2019 | `CSL`     |   [Official](https://github.com/richardaecn/class-balanced-loss)    |
[Striking the right balance with uncertainty](https://openaccess.thecvf.com/content_CVPR_2019/papers/Khan_Striking_the_Right_Balance_With_Uncertainty_CVPR_2019_paper.pdf) | CVPR  | 2019 | `CSL`     |       |
[Feature transfer learning for face recognition with under-represented data](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yin_Feature_Transfer_Learning_for_Face_Recognition_With_Under-Represented_Data_CVPR_2019_paper.pdf) | CVPR  | 2019 | `TL`,`Aug`     |       |
[Unequal-training for deep face recognition with long-tailed noisy data](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhong_Unequal-Training_for_Deep_Face_Recognition_With_Long-Tailed_Noisy_Data_CVPR_2019_paper.pdf) | CVPR  | 2019 | `RL`     |  [Official](https://github.com/zhongyy/Unequal-Training-for-Deep-Face-Recognition-with-Long-Tailed-Noisy-Data)      |
[Large-scale long-tailed recognition in an open world](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Large-Scale_Long-Tailed_Recognition_in_an_Open_World_CVPR_2019_paper.pdf) | CVPR  | 2019 | `RL`     |   [Official](https://github.com/zhmiao/OpenLongTailRecognition-OLTR)     |

### 2018

| Title    | Venue    | Year | Type     | Code     | 
|:-------- |:--------:|:--------:|:--------:|:--------:|
[Large scale fine-grained categorization and domain-specific transfer learning](https://openaccess.thecvf.com/content_cvpr_2018/papers/Cui_Large_Scale_Fine-Grained_CVPR_2018_paper.pdf) | CVPR  | 2018 | `TL`     |   [Official](https://github.com/richardaecn/cvpr18-inaturalist-transfer)      | 

### 2017

| Title    | Venue    | Year | Type     | Code     | 
|:-------- |:--------:|:--------:|:--------:|:--------:|
[Learning to model the tail](https://proceedings.neurips.cc/paper/2017/file/147ebe637038ca50a1265abac8dea181-Paper.pdf) | NeurIPS  | 2017 | `CSL`     |       | 
[Focal loss for dense object detection](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf) | ICCV  | 2017 | `CSL`     |       | 
[Range loss for deep face recognition with long-tailed training data](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_Range_Loss_for_ICCV_2017_paper.pdf) | ICCV  | 2017 | `RL`     |       | 
[Class rectification hard mining for imbalanced deep learning](https://openaccess.thecvf.com/content_ICCV_2017/papers/Dong_Class_Rectification_Hard_ICCV_2017_paper.pdf) | ICCV  | 2017 | `RL`     |       | 

### 2016

| Title    | Venue    | Year | Type     | Code     | 
|:-------- |:--------:|:--------:|:--------:|:--------:|
[Learning deep representation for imbalanced classification](https://openaccess.thecvf.com/content_cvpr_2016/papers/Huang_Learning_Deep_Representation_CVPR_2016_paper.pdf) | CVPR  | 2016 | `Sampling`,`RL`     |       | 
[Factors in finetuning deep model for object detection with long-tail distribution](https://openaccess.thecvf.com/content_cvpr_2016/papers/Ouyang_Factors_in_Finetuning_CVPR_2016_paper.pdf) | CVPR  | 2016 | `CSL`,`RL`     |       | 

## Citing this work 

If this repository is helpful to you, please cite our survey.

```
@article{zhang2021deep,
    title={Deep Long-Tailed Learning: A Survey},
    author={Zhang, Yifan and Kang, Bingyi and Hooi, Bryan and Yan, Shuicheng and Feng, Jiashi},
    journal={arXiv},
    year={2021}
}
```
