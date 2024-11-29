# Awesome Long-Tailed Learning (TPAMI 2023)

  [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
  [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

  We released *[Deep Long-Tailed Learning: A Survey](https://arxiv.org/pdf/2110.04596.pdf)* and **our codebase** to the community. In this survey, we reviewed recent advances in long-tailed learning based on deep neural networks. Existing long-tailed learning studies can be grouped into three main categories (i.e., class re-balancing, information augmentation and module improvement), which can be further classified into nine sub-categories (as shown in the below figure). We also provided empirical analysis for several state-of-the-art methods by evaluating to what extent they address the issue of class imbalance. We concluded the survey by highlighting important applications of deep long-tailed learning and identifying several promising directions for future research. 

  After completing this survey, we decided to release our long-tailed learning resources and codebase, hoping to push the development of the community. If you have any questions or suggestions, please feel free to contact us.

    

  <p align="center">
  <img src="resources/Taxonomy2.png" width=1000>
  </p>
  

  ## 1. Type of Long-tailed Learning

  | Symbol | `Sampling`  |          `CSL`          |       `LA`       |       `TL`        |       `Aug`       |
  | :----- | :---------: | :---------------------: | :--------------: | :---------------: | :---------------: |
  | Type   | Re-sampling | Class-sensitive Learning | Logit Adjustment | Transfer Learning | Data Augmentation |

  | Symbol |          `RL`           |       `CD`        |        `DT`        |    `Ensemble`     |   `other`   |
  | :----- | :---------------------: | :---------------: | :----------------: | :---------------: | :---------: |
  | Type   | Representation Learning | Classifier Design | Decoupled Training | Ensemble Learning | Other Types |

  ## 2. Top-tier Conference Papers (Updated on 2024 December)

  ### 2023

  | Title                                                        |  Venue  | Year |       Type       |                             Code                             |
  | :----------------------------------------------------------- | :-----: | :--: | :--------------: | :----------------------------------------------------------: |
  | [Label-Noise Learning with Intrinsically Long-Tailed Data](https://openaccess.thecvf.com/content/ICCV2023/papers/Lu_Label-Noise_Learning_with_Intrinsically_Long-Tailed_Data_ICCV_2023_paper.pdf) | ICCV | 2023 | `Sampling` |     [Official](https://github.com/Wakings/TABASCO)        |
  | [MDCS: More Diverse Experts with Consistency Self-distillation for Long-tailed Recognition](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_MDCS_More_Diverse_Experts_with_Consistency_Self-distillation_for_Long-tailed_Recognition_ICCV_2023_paper.pdf) | ICCV | 2023 | `Sampling`，`TL`, `Ensemble` |     [Official](https://github.com/fistyee/MDCS)        |
  | [Subclass-balancing Contrastive Learning for Long-tailed Recognition](https://openaccess.thecvf.com/content/ICCV2023/papers/Hou_Subclass-balancing_Contrastive_Learning_for_Long-tailed_Recognition_ICCV_2023_paper.pdf) | ICCV | 2023 | `Sampling`，`RL`|     [Official](https://github.com/JackHck/SBCL)        |
  | [AREA: Adaptive Reweighting via Effective Area for Long-Tailed Classification](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_AREA_Adaptive_Reweighting_via_Effective_Area_for_Long-Tailed_Classification_ICCV_2023_paper.pdf) | ICCV | 2023 | `CSL` |     [Official](https://github.com/xiaohua-chen/AREA)        |
  | [Reconciling Object-Level and Global-Level Objectives for Long-Tail Detection](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Reconciling_Object-Level_and_Global-Level_Objectives_for_Long-Tail_Detection_ICCV_2023_paper.pdf) | ICCV | 2023 | `CSL` |     [Official](https://github.com/EricZsy/ROG)        |
  | [Local and Global Logit Adjustments for Long-Tailed Learning](https://openaccess.thecvf.com/content/ICCV2023/papers/Tao_Local_and_Global_Logit_Adjustments_for_Long-Tailed_Learning_ICCV_2023_paper.pdf) | ICCV | 2023 | `CSL`,`LA`,`Ensemble` |          |
  | [Learning in Imperfect Environment: Multi-Label Classification with Long-Tailed Distribution and Partial Labels](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Learning_in_Imperfect_Environment_Multi-Label_Classification_with_Long-Tailed_Distribution_and_ICCV_2023_paper.pdf) | ICCV | 2023 | `CSL`,`TL` |     [Official](https://github.com/wannature/COMIC)        |
  | [Global Balanced Experts for Federated Long-Tailed Learning](https://openaccess.thecvf.com/content/ICCV2023/papers/Zeng_Global_Balanced_Experts_for_Federated_Long-Tailed_Learning_ICCV_2023_paper.pdf) | ICCV | 2023 | `CSL`, `Ensemble` |     [Official](https://github.com/Spinozaaa/Federated-Long-tailed-Learning)       |
  | [Boosting Long-tailed Object Detection via Step-wise Learning on Smooth-tail Data](https://openaccess.thecvf.com/content/ICCV2023/papers/Dong_Boosting_Long-tailed_Object_Detection_via_Step-wise_Learning_on_Smooth-tail_Data_ICCV_2023_paper.pdf) | ICCV | 2023 | `Ensemble` |         |
  | [Long-tailed recognition by mutual information maximization between latent features and ground-truth labels](https://openreview.net/pdf?id=KqNX6VOqnJ) | ICML | 2023 | `CSL`,`RL` |     [Official](https://github.com/bluecdm/Long-tailed-recognition)        |
  | [Large language models struggle to learn long-tail knowledge](https://openreview.net/pdf?id=sfdKdeczaw) | ICML | 2023 | `Aug` |        |
  | [Feature directions matter: Long-tailed learning via rotated balanced representation](https://openreview.net/pdf?id=dTgxiMW6wr0) | ICML | 2023 | `RL` |        |
  | [Wrapped Cauchy distributed angular softmax for long-tailed visual recognition](https://proceedings.mlr.press/v202/han23a/han23a.pdf) | ICML | 2023 | `RL`,`CD` |  [Official](https://github.com/boranhan/wcdas_code)        | 
  | [Rethinking image super resolution from long-tailed distribution learning perspective](https://openaccess.thecvf.com/content/CVPR2023/papers/Gou_Rethinking_Image_Super_Resolution_From_Long-Tailed_Distribution_Learning_Perspective_CVPR_2023_paper.pdf) | CVPR | 2023 | `CSL` |      |
  | [Transfer knowledge from head to tail: Uncertainty calibration under long-tailed distribution](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Transfer_Knowledge_From_Head_to_Tail_Uncertainty_Calibration_Under_Long-Tailed_CVPR_2023_paper.pdf) | CVPR | 2023 | `CSL`,`TL`  |  [Official](https://github.com/JiahaoChen1/Calibration)        |
  | [Towards realistic long-tailed semi-supervised learning: Consistency is all you need](https://openaccess.thecvf.com/content/CVPR2023/papers/Wei_Towards_Realistic_Long-Tailed_Semi-Supervised_Learning_Consistency_Is_All_You_Need_CVPR_2023_paper.pdf) | CVPR | 2023 | `CSL`,`TL`,`Ensemble`  |  [Official](https://github.com/Gank0078/ACR)        |
  | [Global and local mixture consistency cumulative learning for long-tailed visual recognitions](https://openaccess.thecvf.com/content/CVPR2023/papers/Du_Global_and_Local_Mixture_Consistency_Cumulative_Learning_for_Long-Tailed_Visual_CVPR_2023_paper.pdf) | CVPR | 2023 | `CSL`,`RL` |  [Official](https://github.com/ynu-yangpeng/GLMC)        |
  | [Long-tailed visual recognition via self-heterogeneous integration with knowledge excavation](https://openaccess.thecvf.com/content/CVPR2023/papers/Jin_Long-Tailed_Visual_Recognition_via_Self-Heterogeneous_Integration_With_Knowledge_Excavation_CVPR_2023_paper.pdf) | CVPR | 2023 | `TL`,`Ensemble`  |  [Official](https://github.com/jinyan-06/SHIKE)        |
  | [Balancing logit variation for long-tailed semantic segmentation](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Balancing_Logit_Variation_for_Long-Tailed_Semantic_Segmentation_CVPR_2023_paper.pdf) | CVPR | 2023 | `Aug`  |  [Official](https://github.com/grantword8/BLV)        |
  | [Use your head: Improving long-tail video recognition](https://openaccess.thecvf.com/content/CVPR2023/papers/Perrett_Use_Your_Head_Improving_Long-Tail_Video_Recognition_CVPR_2023_paper.pdf) | CVPR | 2023 | `Aug`  |  [Official](https://github.com/tobyperrett/lmr)        |
  | [FCC: Feature clusters compression for long-tailed visual recognition](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_FCC_Feature_Clusters_Compression_for_Long-Tailed_Visual_Recognition_CVPR_2023_paper.pdf) | CVPR | 2023 | `RL`  |  [Official](https://github.com/lijian16/FCC)        |
  | [FEND: A future enhanced distribution-aware contrastive learning framework for long-tail trajectory prediction](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_FEND_A_Future_Enhanced_Distribution-Aware_Contrastive_Learning_Framework_for_Long-Tail_CVPR_2023_paper.pdf) | CVPR | 2023 | `RL`  |     | 
  | [SuperDisco: Super-class discovery improves visual recognition for the long-tail](https://openaccess.thecvf.com/content/CVPR2023/papers/Du_SuperDisco_Super-Class_Discovery_Improves_Visual_Recognition_for_the_Long-Tail_CVPR_2023_paper.pdf) | CVPR | 2023 | `RL`  |     | 
  | [Class-conditional sharpness-aware minimization for deep long-tailed recognition](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Class-Conditional_Sharpness-Aware_Minimization_for_Deep_Long-Tailed_Recognition_CVPR_2023_paper.pdf) | CVPR | 2023 | `DT`  |  [Official](https://github.com/zzpustc/CC-SAM)        |
  | [Balanced product of calibrated experts for long-tailed recognition](https://openaccess.thecvf.com/content/CVPR2023/papers/Aimar_Balanced_Product_of_Calibrated_Experts_for_Long-Tailed_Recognition_CVPR_2023_paper.pdf) | CVPR | 2023 | `Ensemble`  |  [Official](https://github.com/emasa/BalPoE-CalibratedLT)        |
  | [No one left behind: Improving the worst categories in long-tailed learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Du_No_One_Left_Behind_Improving_the_Worst_Categories_in_Long-Tailed_CVPR_2023_paper.pdf) | CVPR | 2023 | `Ensemble`  |    |
  | [On the effectiveness of out-of-distribution data in self-supervised long-tail learning](https://openreview.net/pdf?id=v8JIQdiN9Sh) | ICLR | 2023 | `Sampling`,`TL`,`Aug`  |  [Official](https://github.com/JianhongBai/COLT)        |
  | [LPT: Long-tailed prompt tuning for image classification](https://openreview.net/pdf?id=8pOVAeo8ie) | ICLR | 2023 | `Sampling`,`TL`,`Other`  |  [Official](https://github.com/DongSky/LPT)        |
  | [Long-tailed partial label learning via dynamic rebalancing](https://openreview.net/pdf?id=sXfWoK4KvSW) | ICLR | 2023 | `CSL`  |  [Official](https://github.com/MediaBrain-SJTU/RECORDS-LTPLL)        |
  | [Delving into semantic scale imbalance](https://openreview.net/pdf?id=07tc5kKRIo) | ICLR | 2023 | `CSL`,`RL` |       |
  | [INPL: Pseudo-labeling the inliers first for imbalanced semi-supervised learning](https://openreview.net/pdf?id=m6ahb1mpwwX) | ICLR | 2023 | `TL` |       |
  | [CUDA: Curriculum of data augmentation for long-tailed recognition](https://openreview.net/pdf?id=RgUPdudkWlN) | ICLR | 2023 | `Aug` |  [Official](https://github.com/JianhongBai/COLT)        |
  | [Long-tailed learning requires feature learning](https://openreview.net/pdf?id=S-h1oFv-mq) | ICLR | 2023 | `RL`  |     | 
  | [Decoupled training for long-tailed classification with stochastic representations](https://openreview.net/pdf?id=bcYZwYo-0t) | ICLR | 2023 | `RL`,`DT` |      |



  ### 2022

  | Title                                                        |  Venue  | Year |       Type       |                             Code                             |
  | :----------------------------------------------------------- | :-----: | :--: | :--------------: | :----------------------------------------------------------: |
  | [Self-supervised aggregation of diverse experts for test-agnostic long-tailed recognition](https://openreview.net/pdf?id=m7CmxlpHTiu) | NeurIPS | 2022 | `CSL`,`Ensemble` |    [Official](https://github.com/Vanint/SADE-AgnosticLT)     |
  | [SoLar: Sinkhorn label refinery for imbalanced partial-label learning](https://openreview.net/pdf?id=wUUutywJY6) | NeurIPS | 2022 | `CSL` |    [Official](https://github.com/hbzju/SoLar)     |
  | [Do we really need a learnable classifier at the end of deep neural network?](https://openreview.net/pdf?id=A6EmxI3_Xc) | NeurIPS | 2022 |    `RL`,`CD`     |                                                              |
  | [Maximum class separation as inductive bias in one matrix](https://openreview.net/pdf?id=MbVS6BuJ3ql) | NeurIPS | 2022 |   `CD`     |         [Official](https://github.com/tkasarla/max-separation-as-inductive-bias)     |                                                       |
  | [Escaping saddle points for effective generalization on class-imbalanced data](https://openreview.net/pdf?id=9DYKrsFSU2) | NeurIPS | 2022 | `other` |    [Official](https://github.com/val-iisc/Saddle-LongTail)     |                                           |
  | [Breadcrumbs: Adversarial class-balanced sampling for long-tailed recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840628.pdf) | ECCV | 2022 | `Sampling`,`Aug`,`DT` |    [Official](https://github.com/BoLiu-SVCL/Breadcrumbs)     |
  | [Constructing balance from imbalance for long-tailed image recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800036.pdf) | ECCV | 2022 | `Sampling`,`RL` |    [Official](https://github.com/silicx/DLSA)     |
  | [Tackling long-tailed category distribution under domain shifts](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830706.pdf) | ECCV | 2022 | `CSL`,`Aug`,`RL` |    [Official](https://github.com/guxiao0822/lt-ds)     |
  | [Improving GANs for long-tailed data through group spectral regularization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750423.pdf) | ECCV | 2022 | `CSL`,`Other` |    [Official](https://github.com/val-iisc/gSRGAN)     |
  | [Learning class-wise visual-linguistic representation for long-tailed visual recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850072.pdf) | ECCV | 2022 | `TL`,`RL` |    [Official](https://github.com/ChangyaoTian/VL-LTR)     |
  | [Learning with free object segments for long-tailed instance segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700648.pdf) | ECCV | 2022 | `Aug` |       |
  | [SAFA: Sample-adaptive feature augmentation for long-tailed image classification](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840578.pdf) | ECCV | 2022 | `Aug`,`RL` |    | 
  | [On multi-domain long-tailed recognition, imbalanced domain generalization, and beyond](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800054.pdf) | ECCV | 2022 | `RL`|    [Official](https://github.com/YyzHarry/multi-domain-imbalance)     | 
  | [Invariant feature learning for generalized long-tailed classification](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840698.pdf) | ECCV | 2022 | `RL`|    [Official](https://github.com/KaihuaTang/Generalized-Long-Tailed-Benchmarks.pytorch)     |
  | [Towards calibrated hyper-sphere representation via distribution overlap coefficient for long-tailed learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840176.pdf) | ECCV | 2022 | `RL`,`CD` |    [Official](https://github.com/SiLangWHL/vMF-OP)     |
  | [Long-tailed instance segmentation using Gumbel optimized loss](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700349.pdf) | ECCV | 2022 | `CD` |    [Official](https://github.com/kostas1515/GOL)     |
  | [Long-tailed class incremental learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930486.pdf) | ECCV | 2022 | `DT` |    [Official](https://github.com/xialeiliu/Long-Tailed-CIL)     |
  | [Identifying hard noise in long-tailed sample distribution](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860725.pdf) | ECCV | 2022 | `Other` |    [Official](https://github.com/yxymessi/H2E-Framework)     |
  | [Relieving long-tailed instance segmentation via pairwise class balance](https://arxiv.org/pdf/2201.02784.pdf) |  CVPR   | 2022 |      `CSL`       |      [Official](https://github.com/megvii-research/PCB)      |
  | [The majority can help the minority: Context-rich minority oversampling for long-tailed classification](https://arxiv.org/pdf/2112.00412.pdf) |  CVPR   | 2022 |    `TL`,`Aug`    |         [Official](https://github.com/naver-ai/cmo)          |
  | [Long-tail recognition via compositional knowledge transfer](https://arxiv.org/pdf/2112.06741.pdf) |  CVPR   | 2022 |    `TL`,`RL`     |                                                              |
  | [BatchFormer: Learning to explore sample relationships for robust representation learning](https://arxiv.org/pdf/2203.01522.pdf) |  CVPR   | 2022 |    `TL`,`RL`     |      [Official](https://github.com/zhihou7/BatchFormer)      |
  | [Nested collaborative learning for long-tailed visual recognition](https://arxiv.org/pdf/2203.15359.pdf) |  CVPR   | 2022 | `RL`,`Ensemble`  |        [Official](https://github.com/Bazinga699/NCL)         |
  | [Long-tailed recognition via weight balancing](https://arxiv.org/pdf/2203.14197.pdf) |  CVPR   | 2022 |       `DT`       | [Official](https://github.com/ShadeAlsha/LTR-weight-balancing) |
  | [Class-balanced pixel-level self-labeling for domain adaptive semantic segmentation](https://arxiv.org/pdf/2203.09744.pdf) |  CVPR   | 2022 |     `other`      |          [Official](https://github.com/lslrh/CPSL)           |
  | [Killing two birds with one stone: Efficient and robust training of face recognition CNNs by partial FC](https://arxiv.org/pdf/2203.15565.pdf) |  CVPR   | 2022 |     `other`      | [Official](https://github.com/deepinsight/insightface/tree/master/recognition) |
  | [Optimal transport for long-tailed recognition with learnable cost matrix](https://openreview.net/pdf?id=t98k9ePQQpn) |  ICLR   | 2022 |       `LA`       |                                                              |
  | [Do deep networks transfer invariances across classes?](https://openreview.net/pdf?id=Fn7i_r5rR0q) |  ICLR   | 2022 |    `TL`,`Aug`    | [Official](https://github.com/AllanYangZhou/generative-invariance-transfer) |
  | [Self-supervised learning is more robust to dataset imbalance](https://openreview.net/pdf?id=4AZz9osqrar) |  ICLR   | 2022 |       `RL`       |                                                              |

  ### 2021

  | Title                                                        |  Venue  | Year |             Type             |                             Code                             |
  | :----------------------------------------------------------- | :-----: | :--: | :--------------------------: | :----------------------------------------------------------: |
  | [Improving contrastive learning on imbalanced seed data via open-world sampling](https://openreview.net/pdf?id=EIfV-XAggKo) | NeurIPS | 2021 |    `Sampling`,`TL`, `DC`     |        [Official](https://github.com/VITA-Group/MAK)         |
  | [Semi-supervised semantic segmentation via adaptive equalization learning](https://papers.nips.cc/paper/2021/file/b98249b38337c5088bbc660d8f872d6a-Paper.pdf) | NeurIPS | 2021 | `Sampling`,`CSL`,`TL`, `Aug` |      [Official](https://github.com/hzhupku/SemiSeg-AEL)      |
  | [On model calibration for long-tailed object detection and instance segmentation](https://proceedings.neurips.cc/paper/2021/file/14ad095ecc1c3e1b87f3c522836e9158-Paper.pdf) | NeurIPS | 2021 |             `LA`             |         [Official](https://github.com/tydpan/NorCal)         |
  | [Label-imbalanced and group-sensitive classification under overparameterization](https://openreview.net/pdf?id=UZm2IQhgIyB) | NeurIPS | 2021 |             `LA`             |                                                              |
  | [Towards calibrated model for long-tailed visual recognition from prior perspective](https://papers.nips.cc/paper/2021/file/39ae2ed11b14a4ccb41d35e9d1ba5d11-Paper.pdf) | NeurIPS | 2021 |         `Aug`, `RL`          |     [Official](https://github.com/XuZhengzhuo/Prior-LT)      |
  | [Supercharging imbalanced data learning with energy-based contrastive representation transfer](https://papers.nips.cc/paper/2021/file/b151ce4935a3c2807e1dd9963eda16d8-Paper.pdf) | NeurIPS | 2021 |      `Aug`, `TL`, `RL`       |         [Official](https://github.com/ZidiXiu/ECRT)          |
  | [VideoLT: Large-scale long-tailed video recognition](https://arxiv.org/pdf/2105.02668.pdf) |  ICCV   | 2021 |          `Sampling`          |       [Official](https://github.com/17Skye17/VideoLT)        |
  | [Exploring classification equilibrium in long-tailed object detection](https://arxiv.org/pdf/2108.07507.pdf) |  ICCV   | 2021 |       `Sampling`,`CSL`       |          [Official](https://github.com/fcjian/LOCE)          |
  | [GistNet: a geometric structure transfer network for long-tailed recognition](https://arxiv.org/pdf/2105.00131.pdf) |  ICCV   | 2021 |    `Sampling`,`TL`, `DC`     |                                                              |
  | [FASA: Feature augmentation and sampling adaptation for long-tailed instance segmentation](https://arxiv.org/pdf/2102.12867.pdf) |  ICCV   | 2021 |       `Sampling`,`CSL`       |                                                              |
  | [ACE: Ally complementary experts for solving long-tailed recognition in one-shot](https://arxiv.org/pdf/2108.02385.pdf) |  ICCV   | 2021 |    `Sampling`,`Ensemble`     | [Official](https://github.com/jrcai/ACE?utm_source=catalyzex.com) |
  | [Influence-Balanced Loss for Imbalanced Visual Classification](https://arxiv.org/pdf/2110.02444.pdf) |  ICCV   | 2021 |            `CSL`             |        [Official](https://github.com/pseulki/IB-Loss)        |
  | [Re-distributing biased pseudo labels for semi-supervised semantic segmentation: A baseline investigation](https://arxiv.org/pdf/2107.11279.pdf) |  ICCV   | 2021 |             `TL`             |         [Official](https://github.com/CVMI-Lab/DARS)         |
  | [Self supervision to distillation for long-tailed visual recognition](https://arxiv.org/pdf/2109.04075.pdf) |  ICCV   | 2021 |             `TL`             |        [Official](https://github.com/MCG-NJU/SSD-LT)         |
  | [Distilling virtual examples for long-tailed recognition](https://cs.nju.edu.cn/wujx/paper/ICCV2021_DiVE.pdf) |  ICCV   | 2021 |             `TL`             |                                                              |
  | [MosaicOS: A simple and effective use of object-centric images for long-tailed object detection](https://arxiv.org/pdf/2102.08884.pdf) |  ICCV   | 2021 |             `TL`             |     [Official](https://github.com/czhang0528/MosaicOS/)      |
  | [Parametric contrastive learning](https://arxiv.org/pdf/2107.12028.pdf) |  ICCV   | 2021 |             `RL`             | [Official](https://github.com/dvlab-research/Parametric-Contrastive-Learning) |
  | [Distributional robustness loss for long-tail learning](https://arxiv.org/pdf/2104.03066.pdf) |  ICCV   | 2021 |             `RL`             |       [Official](https://github.com/dvirsamuel/DRO-LT)       |
  | [Learning of visual relations: The devil is in the tails](https://arxiv.org/pdf/2108.09668.pdf) |  ICCV   | 2021 |             `DT`             |                                                              |
  | [Image-Level or Object-Level? A Tale of Two Resampling Strategies for Long-Tailed Detection](https://arxiv.org/pdf/2104.05702.pdf) |  ICML   | 2021 |          `Sampling`          |          [Official](https://github.com/NVlabs/RIO)           |
  | [Self-Damaging Contrastive Learning](https://arxiv.org/pdf/2106.02990.pdf) |  ICML   | 2021 |          `TL`,`RL`           |       [Official](https://github.com/VITA-Group/SDCLR)        |
  | [Delving into deep imbalanced regression](https://arxiv.org/pdf/2102.09554.pdf) |  ICML   | 2021 |           `Other`            | [Official](https://github.com/YyzHarry/imbalanced-regression) |
  | [Long-tailed multi-label visual recognition by collaborative training on uniform and re-balanced samplings](https://openaccess.thecvf.com/content/CVPR2021/papers/Guo_Long-Tailed_Multi-Label_Visual_Recognition_by_Collaborative_Training_on_Uniform_and_CVPR_2021_paper.pdf) |  CVPR   | 2021 |    `Sampling`,`Ensemble`     |                                                              |
  | [Equalization loss v2: A new gradient balance approach for long-tailed object detection](https://openaccess.thecvf.com/content/CVPR2021/papers/Tan_Equalization_Loss_v2_A_New_Gradient_Balance_Approach_for_Long-Tailed_CVPR_2021_paper.pdf) |  CVPR   | 2021 |            `CSL`             |       [Official](https://github.com/tztztztztz/eqlv2)        |
  | [Seesaw loss for long-tailed instance segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Seesaw_Loss_for_Long-Tailed_Instance_Segmentation_CVPR_2021_paper.pdf) |  CVPR   | 2021 |            `CSL`             |    [Official](https://github.com/open-mmlab/mmdetection)     |
  | [Adaptive class suppression loss for long-tail object detection](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Adaptive_Class_Suppression_Loss_for_Long-Tail_Object_Detection_CVPR_2021_paper.pdf) |  CVPR   | 2021 |            `CSL`             |      [Official](https://github.com/CASIA-IVA-Lab/ACSL)       |
  | [PML: Progressive margin loss for long-tailed age classification](https://openaccess.thecvf.com/content/CVPR2021/papers/Deng_PML_Progressive_Margin_Loss_for_Long-Tailed_Age_Classification_CVPR_2021_paper.pdf) |  CVPR   | 2021 |            `CSL`             |                                                              |
  | [Disentangling label distribution for long-tailed visual recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/Hong_Disentangling_Label_Distribution_for_Long-Tailed_Visual_Recognition_CVPR_2021_paper.pdf) |  CVPR   | 2021 |          `CSL`,`LA`          |       [Official](https://github.com/hyperconnect/LADE)       |
  | [Adversarial robustness under long-tailed distribution](https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_Adversarial_Robustness_Under_Long-Tailed_Distribution_CVPR_2021_paper.pdf) |  CVPR   | 2021 |       `CSL`,`LA`,`CD`        | [Official](https://github.com/wutong16/Adversarial_Long-Tail) |
  | [Distribution alignment: A unified framework for long-tail visual recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Distribution_Alignment_A_Unified_Framework_for_Long-Tail_Visual_Recognition_CVPR_2021_paper.pdf) |  CVPR   | 2021 |       `CSL`,`LA`,`DT`        | [Official](https://github.com/Megvii-BaseDetection/DisAlign) |
  | [Improving calibration for long-tailed recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhong_Improving_Calibration_for_Long-Tailed_Recognition_CVPR_2021_paper.pdf) |  CVPR   | 2021 |       `CSL`,`Aug`,`DT`       |     [Official](https://github.com/dvlab-research/MiSLAS)     |
  | [CReST: A class-rebalancing self-training framework for imbalanced semi-supervised learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Wei_CReST_A_Class-Rebalancing_Self-Training_Framework_for_Imbalanced_Semi-Supervised_Learning_CVPR_2021_paper.pdf) |  CVPR   | 2021 |             `TL`             |     [Official](https://github.com/google-research/crest)     |
  | [Conceptual 12M: Pushing web-scale image-text pre-training to recognize long-tail visual concepts](https://openaccess.thecvf.com/content/CVPR2021/papers/Changpinyo_Conceptual_12M_Pushing_Web-Scale_Image-Text_Pre-Training_To_Recognize_Long-Tail_Visual_CVPR_2021_paper.pdf) |  CVPR   | 2021 |             `TL`             | [Official](https://github.com/google-research-datasets/conceptual-12m) |
  | [RSG: A simple but effective module for learning imbalanced datasets](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Wang_RSG_A_Simple_CVPR_2021_supplemental.pdf) |  CVPR   | 2021 |          `TL`,`Aug`          |        [Official](https://github.com/Jianf-Wang/RSG)         |
  | [MetaSAug: Meta semantic augmentation for long-tailed visual recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_MetaSAug_Meta_Semantic_Augmentation_for_Long-Tailed_Visual_Recognition_CVPR_2021_paper.pdf) |  CVPR   | 2021 |            `Aug`             |        [Official](https://github.com/BIT-DA/MetaSAug)        |
  | [Contrastive learning based hybrid networks for long-tailed image classification](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Contrastive_Learning_Based_Hybrid_Networks_for_Long-Tailed_Image_Classification_CVPR_2021_paper.pdf) |  CVPR   | 2021 |             `RL`             |                                                              |
  | [Unsupervised discovery of the long-tail in instance segmentation using hierarchical self-supervision](https://openaccess.thecvf.com/content/CVPR2021/papers/Weng_Unsupervised_Discovery_of_the_Long-Tail_in_Instance_Segmentation_Using_Hierarchical_CVPR_2021_paper.pdf) |  CVPR   | 2021 |             `RL`             |                                                              |
  | [Long-tail learning via logit adjustment](https://openreview.net/pdf?id=37nvvqkCo5) |  ICLR   | 2021 |             `LA`             | [Official](https://github.com/google-research/google-research/tree/master/logit_adjustment) |
  | [Long-tailed recognition by routing diverse distribution-aware experts](https://openreview.net/pdf?id=D9I3drBz4UC) |  ICLR   | 2021 |       `TL`,`Ensemble`        | [Official](https://github.com/frank-xwang/RIDE-LongTailRecognition) |
  | [Exploring balanced feature spaces for representation learning](https://openreview.net/pdf?id=OqtLIabPTit) |  ICLR   | 2021 |          `RL`,`DT`           |                                                              |

  ### 2020

  | Title                                                        |  Venue  | Year |              Type               |                             Code                             |
  | :----------------------------------------------------------- | :-----: | :--: | :-----------------------------: | :----------------------------------------------------------: |
  | [Balanced meta-softmax for long-taield visual recognition](https://proceedings.neurips.cc/paper/2020/file/2ba61cc3a8f44143e1f2f13b2b729ab3-Paper.pdf) | NeurIPS | 2020 |        `Sampling`,`CSL`         | [Official](https://github.com/jiawei-ren/BalancedMetaSoftmax) |
  | [Posterior recalibration for imbalanced datasets](https://proceedings.neurips.cc/paper/2020/file/5ca359ab1e9e3b9c478459944a2d9ca5-Paper.pdf) | NeurIPS | 2020 |              `LA`               |        [Official](https://github.com/GT-RIPL/UNO-IC)         |
  | [Long-tailed classification by keeping the good and removing the bad momentum causal effect](https://proceedings.neurips.cc/paper/2020/file/1091660f3dff84fd648efe31391c5524-Paper.pdf) | NeurIPS | 2020 |            `LA`,`CD`            | [Official](https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch) |
  | [Rethinking the value of labels for improving classimbalanced learning](https://proceedings.neurips.cc/paper/2020/file/e025b6279c1b88d3ec0eca6fcb6e6280-Paper.pdf) | NeurIPS | 2020 |            `TL`,`RL`            | [Official](https://github.com/YyzHarry/imbalanced-semi-self) |
  | [The devil is in classification: A simple framework for long-tail instance segmentation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590715.pdf) |  ECCV   | 2020 |   `Sampling`,`DT`,`Ensemble`    |        [Official](https://github.com/twangnh/SimCal)         |
  | [Imbalanced continual learning with partitioning reservoir sampling](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580409.pdf) |  ECCV   | 2020 |           `Sampling`            |          [Official](https://github.com/cdjkim/PRS)           |
  | [Distribution-balanced loss for multi-label classification in long-tailed datasets](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490154.pdf) |  ECCV   | 2020 |              `CSL`              | [Official](https://github.com/wutong16/DistributionBalancedLoss) |
  | [Feature space augmentation for long-tailed data](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740681.pdf) |  ECCV   | 2020 |         `TL`,`Aug`,`DT`         |                                                              |
  | [Learning from multiple experts: Self-paced knowledge distillation for long-tailed classification](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500239.pdf) |  ECCV   | 2020 |         `TL`,`Ensemble`         |        [Official](https://github.com/xiangly55/LFME)         |
  | [Solving long-tailed recognition with deep realistic taxonomic classifier](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530171.pdf) |  ECCV   | 2020 |              `CD`               |       [Official](https://github.com/gina9726/Deep-RTC)       |
  | [Learning to segment the tail](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hu_Learning_to_Segment_the_Tail_CVPR_2020_paper.pdf) |  CVPR   | 2020 |         `Sampling`,`TL`         |     [Official](https://github.com/JoyHuYY1412/LST_LVIS)      |
  | [BBN: Bilateral-branch network with cumulative learning for long-tailed visual recognition](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_BBN_Bilateral-Branch_Network_With_Cumulative_Learning_for_Long-Tailed_Visual_Recognition_CVPR_2020_paper.pdf) |  CVPR   | 2020 |      `Sampling`,`Ensemble`      |      [Official](https://github.com/Megvii-Nanjing/BBN)       |
  | [Overcoming classifier imbalance for long-tail object detection with balanced group softmax](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Overcoming_Classifier_Imbalance_for_Long-Tail_Object_Detection_With_Balanced_Group_CVPR_2020_paper.pdf) |  CVPR   | 2020 |      `Sampling`,`Ensemble`      | [Official](https://github.com/FishYuLi/BalancedGroupSoftmax) |
  | [Rethinking class-balanced methods for long-tailed visual recognition from a domain adaptation perspective](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jamal_Rethinking_Class-Balanced_Methods_for_Long-Tailed_Visual_Recognition_From_a_Domain_CVPR_2020_paper.pdf) |  CVPR   | 2020 |              `CSL`              |   [Official](https://github.com/abdullahjamal/Longtail_DA)   |
  | [Equalization loss for long-tailed object recognition](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tan_Equalization_Loss_for_Long-Tailed_Object_Recognition_CVPR_2020_paper.pdf) |  CVPR   | 2020 |              `CSL`              |       [Official](https://github.com/tztztztztz/eqlv2)        |
  | [Domain balancing: Face recognition on long-tailed domains](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cao_Domain_Balancing_Face_Recognition_on_Long-Tailed_Domains_CVPR_2020_paper.pdf) |  CVPR   | 2020 |              `CSL`              |                                                              |
  | [M2m: Imbalanced classification via majorto-minor translation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_M2m_Imbalanced_Classification_via_Major-to-Minor_Translation_CVPR_2020_paper.pdf) |  CVPR   | 2020 |           `TL`,`Aug`            |          [Official](https://github.com/alinlab/M2m)          |
  | [Deep representation learning on long-tailed data: A learnable embedding augmentation perspective](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Deep_Representation_Learning_on_Long-Tailed_Data_A_Learnable_Embedding_Augmentation_CVPR_2020_paper.pdf) |  CVPR   | 2020 |         `TL`,`Aug`,`RL`         |                                                              |
  | [Inflated episodic memory with region self-attention for long-tailed visual recognition](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhu_Inflated_Episodic_Memory_With_Region_Self-Attention_for_Long-Tailed_Visual_Recognition_CVPR_2020_paper.pdf) |  CVPR   | 2020 |              `RL`               |                                                              |
  | [Decoupling representation and classifier for long-tailed recognition](https://openreview.net/pdf?id=r1gRTCVFvB) |  ICLR   | 2020 | `Sampling`,`CSL`,`RL`,`CD`,`DT` | [Official](https://github.com/facebookresearch/classifier-balancing) |


  ### 2019

  | Title                                                        |  Venue  | Year |    Type    |                             Code                             |
  | :----------------------------------------------------------- | :-----: | :--: | :--------: | :----------------------------------------------------------: |
  | [Meta-weight-net: Learning an explicit mapping for sample weighting](https://proceedings.neurips.cc/paper/2019/file/e58cc5ca94270acaceed13bc82dfedf7-Paper.pdf) | NeurIPS | 2019 |   `CSL`    |  [Official](https://github.com/xjtushujun/meta-weight-net)   |
  | [Learning imbalanced datasets with label-distribution-aware margin loss](https://proceedings.neurips.cc/paper/2019/file/621461af90cadfdaf0e8d4cc25129f91-Paper.pdf) | NeurIPS | 2019 |   `CSL`    |        [Official](https://github.com/kaidic/LDAM-DRW)        |
  | [Dynamic curriculum learning for imbalanced data classification](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Dynamic_Curriculum_Learning_for_Imbalanced_Data_Classification_ICCV_2019_paper.pdf) |  ICCV   | 2019 | `Sampling` |                                                              |
  | [Class-balanced loss based on effective number of samples](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf) |  CVPR   | 2019 |   `CSL`    | [Official](https://github.com/richardaecn/class-balanced-loss) |
  | [Striking the right balance with uncertainty](https://openaccess.thecvf.com/content_CVPR_2019/papers/Khan_Striking_the_Right_Balance_With_Uncertainty_CVPR_2019_paper.pdf) |  CVPR   | 2019 |   `CSL`    |                                                              |
  | [Feature transfer learning for face recognition with under-represented data](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yin_Feature_Transfer_Learning_for_Face_Recognition_With_Under-Represented_Data_CVPR_2019_paper.pdf) |  CVPR   | 2019 | `TL`,`Aug` |                                                              |
  | [Unequal-training for deep face recognition with long-tailed noisy data](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhong_Unequal-Training_for_Deep_Face_Recognition_With_Long-Tailed_Noisy_Data_CVPR_2019_paper.pdf) |  CVPR   | 2019 |    `RL`    | [Official](https://github.com/zhongyy/Unequal-Training-for-Deep-Face-Recognition-with-Long-Tailed-Noisy-Data) |
  | [Large-scale long-tailed recognition in an open world](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Large-Scale_Long-Tailed_Recognition_in_an_Open_World_CVPR_2019_paper.pdf) |  CVPR   | 2019 |    `RL`    | [Official](https://github.com/zhmiao/OpenLongTailRecognition-OLTR) |

  ### 2018

  | Title                                                        | Venue | Year | Type |                             Code                             |
  | :----------------------------------------------------------- | :---: | :--: | :--: | :----------------------------------------------------------: |
  | [Large scale fine-grained categorization and domain-specific transfer learning](https://openaccess.thecvf.com/content_cvpr_2018/papers/Cui_Large_Scale_Fine-Grained_CVPR_2018_paper.pdf) | CVPR  | 2018 | `TL` | [Official](https://github.com/richardaecn/cvpr18-inaturalist-transfer) |

  ### 2017

  | Title                                                        |  Venue  | Year | Type  | Code |
  | :----------------------------------------------------------- | :-----: | :--: | :---: | :--: |
  | [Learning to model the tail](https://proceedings.neurips.cc/paper/2017/file/147ebe637038ca50a1265abac8dea181-Paper.pdf) | NeurIPS | 2017 | `CSL` |      |
  | [Focal loss for dense object detection](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf) |  ICCV   | 2017 | `CSL` |      |
  | [Range loss for deep face recognition with long-tailed training data](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_Range_Loss_for_ICCV_2017_paper.pdf) |  ICCV   | 2017 | `RL`  |      |
  | [Class rectification hard mining for imbalanced deep learning](https://openaccess.thecvf.com/content_ICCV_2017/papers/Dong_Class_Rectification_Hard_ICCV_2017_paper.pdf) |  ICCV   | 2017 | `RL`  |      |

  ### 2016

  | Title                                                        | Venue | Year |      Type       | Code |
  | :----------------------------------------------------------- | :---: | :--: | :-------------: | :--: |
  | [Learning deep representation for imbalanced classification](https://openaccess.thecvf.com/content_cvpr_2016/papers/Huang_Learning_Deep_Representation_CVPR_2016_paper.pdf) | CVPR  | 2016 | `Sampling`,`RL` |      |
  | [Factors in finetuning deep model for object detection with long-tail distribution](https://openaccess.thecvf.com/content_cvpr_2016/papers/Ouyang_Factors_in_Finetuning_CVPR_2016_paper.pdf) | CVPR  | 2016 |   `CSL`,`RL`    |      |

  ## 3. Benchmark Datasets

  | Dataset          |      Long-tailed Task      | # Class | # Training data | # Test data |
  | :--------------- | :------------------------: | :-----: | :-------------: | :---------: |
  | ImageNet-LT      |       Classification       |  1,000  |     115,846     |   50,000    |
  | CIFAR100-LT      |       Classification       |   100   |     50,000      |   10,000    |
  | Places-LT        |       Classification       |   365   |     62,500      |   36,500    |
  | iNaturalist 2018 |       Classification       |  8,142  |     437,513     |   24,426    |
  | LVIS v0.5        | Detection and Segmentation |  1,230  |     57,000      |   20,000    |
  | LVIS v1          | Detection and Segmentation |  1,203  |     100,000     |   19,800    |
  | VOC-LT           | Multi-label Classification |   20    |      1,142      |    4,952    |
  | COCO-LT          | Multi-label Classification |   80    |      1,909      |    5,000    |
  | VideoLT          |    Video Classification    |  1,004  |     179,352     |   25,622    |

  ## 4. Our codebase

  * To use our codebase, please install requirements: 
    ```
    pip install -r requirements.txt
    ```
  * Hardware requirements: 4 GPUs with >= 23G GPU RAM are recommended.  
  * ImageNet-LT dataset: please download ImageNet-1K dataset, and put it to the ./data file.
    ```
    data
    └──ImageNet
        ├── train
        └── val
    ```
  * Softmax:
    ```
    cd ./Main-codebase 
    Training: python3 main.py --seed 1 --cfg config/ImageNet_LT/ce.yaml  --exp_name imagenet/CE  --gpu 0,1,2,3 
    ```
  * Weighted Softmax:
    ```
    cd ./Main-codebase 
    Training: python3 main.py --seed 1 --cfg config/ImageNet_LT/weighted_ce.yaml  --exp_name imagenet/weighted_ce  --gpu 0,1,2,3
    ```
  * ESQL (Equalization loss):
    ```
    cd ./Main-codebase 
    Training: python3 main.py --seed 1 --cfg config/ImageNet_LT/seql.yaml  --exp_name imagenet/seql  --gpu 0,1,2,3
    ```
  * Balanced Softmax:
    ```
    cd ./Main-codebase 
    Training: python3 main.py --seed 1 --cfg config/ImageNet_LT/balanced_softmax.yaml  --exp_name imagenet/BS  --gpu 0,1,2,3
    ```
  * LADE:
    ```
    cd ./Main-codebase 
    Training: python3 main.py --seed 1 --cfg config/ImageNet_LT/lade.yaml  --exp_name imagenet/LADE  --gpu 0,1,2,3
    ```
  * De-confound (Casual):
    ```
    cd ./Main-codebase 
    Training: python3 main.py --seed 1 --cfg config/ImageNet_LT/causal.yaml  --exp_name imagenet/causal --remine_lambda 0.1 --alpha 0.005 --gpu 0,1,2,3
    ```
  * Decouple (IB-CRT):
    ```
    cd ./Main-codebase 
    Training stage 1: python3 main.py --seed 1 --cfg config/ImageNet_LT/ce.yaml  --exp_name imagenet/CE  --gpu 0,1,2,3 
    Training stage 2: python3  main.py --cfg ./config/ImageNet_LT/cls_crt.yaml --model_dir exp_results/imagenet/CE/final_model_checkpoint.pth  --gpu 0,1,2,3 
    ```
  * MiSLAS:
    ```
    cd ./MiSLAS-codebase
    Training stage 1: CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_stage1.py --cfg config/imagenet/imagenet_resnext50_stage1_mixup.yaml
    Training stage 2: CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_stage2.py --cfg config/imagenet/imagenet_resnext50_stage2_mislas.yaml resume checkpoint_path
    Evalutation: CUDA_VISIBLE_DEVICES=0  python3 eval.py --cfg ./config/imagenet/imagenet_resnext50_stage2_mislas.yaml  resume checkpoint_path_stage2
    ```
  * RSG:
    ```
    cd ./RSG-codebase
    Training: python3 imagenet_lt_train.py 
    Evalutation: python3 imagenet_lt_test.py 
    ```
  * ResLT:
    ```
    cd ./ResLT-codebase
    Training: CUDA_VISIBLE_DEVICES=0,1,2,3 bash sh/X50.sh
    Evalutation: CUDA_VISIBLE_DEVICES=0 bash sh/X50_eval.sh
    # The test performance can be found in the log file.
    ```
  * PaCo:
    ```
    cd ./PaCo-codebase
    Training: CUDA_VISIBLE_DEVICES=0,1,2,3 bash sh/ImageNetLT_train_X50.sh
    Evalutation: CUDA_VISIBLE_DEVICES=0 bash sh/ImageNetLT_eval_X50.sh
    # The test performance can be found in the log file.
    ```
  * LDAM:
    ```
    cd ./Ensemble-codebase 
    Training: CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py -c ./configs/config_imagenet_lt_resnext50_ldam.json
    Evalutation: CUDA_VISIBLE_DEVICES=0 python3 test.py -r checkpoint_path
    ```
  * RIDE:
    ```
    cd ./Ensemble-codebase 
    Training: CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py -c ./configs/config_imagenet_lt_resnext50_ride.json
    Evalutation: CUDA_VISIBLE_DEVICES=0 python3 test.py -r checkpoint_path
    ```
  * SADE:
    ```
    cd ./Ensemble-codebase 
    Training: CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py -c ./configs/config_imagenet_lt_resnext50_sade.json
    Evalutation: CUDA_VISIBLE_DEVICES=0 python3 test.py -r checkpoint_path
    ```

  ## 5. Empirical Studies

  ### (1) Long-tailed benchmarking performance

  * We evaluate several state-of-the-art methods on ImageNet-LT to see to what extent they handle class imbalance via new evaluation metrics, i.e., UA (upper bound accuracy) and RA (relative accuracy). We categorize these methods based on class re-balancing (CR), information augmentation (IA) and module improvement (MI). 

  <p align="center">
  <img src="resources/Fig1.png" width=900>
  </p>
  

  * Almost all long-tailed methods perform better than the Softmax baseline in terms of accuracy, which demonstrates the effectiveness of long-tailed learning. 
  * Training with 200 epochs leads to better performance for most long-tailed methods, since sufficient  training enables deep models to fit data better and learn better image representations.
  * In addition to accuracy, we also evaluate long-tailed  methods based on UA and RA. For the methods that have higher  UA, the performance gain  comes  not only  from the alleviation of class imbalance, but also from other factors, like data augmentation or better network architectures. Therefore, simply using accuracy for evaluation is not accurate enough, while our proposed RA metric provides a good complement, since it alleviates the influences of factors apart from class imbalance. 
  * For example, MiSLAS, based on data mixup, has higher accuracy than Balanced Sofmtax under 90 training epochs, but it also has higher UA. As a result, the relative accuracy of  MiSLAS is lower than Balanced Sofmtax, which means that Balanced Sofmtax alleviates class imbalance better than MiSLAS under 90 training epochs.
  * Although some recent high-accuracy methods   have  lower RA, the overall development trend of long-tailed learning is still positive, as shown in the below figure.


  <p align="center">
  <img src="resources/Fig2.png" width=900>
  </p>
  

  * The current state-of-the-art long-tailed method in terms of both accuracy and RA is SADE (ensemble-based method). 

  ### (2) More discussions on cost-sensitive losses

  * We further evaluate the performance of different cost-sensitive learning losses based on the decoupled training scheme.
  * Decoupled training, compared to joint training, can further improve  the   overall performance  of most cost-sensitive learning methods apart from balanced softmax (BS).
  * Although BS outperofmrs other cost-sensitive losses under one-stage training, they perform comparably under decoupled training. This implies that although these cost-sensitive losses perform differently under joint training, they essentially learn similar  quality of feature representations. 

  <p align="center">
  <img src="resources/Fig3.png" width=500>
  </p>
  


  ## 5. Citation

  If this repository is helpful to you, please cite our survey.

  ```
  @article{zhang2023deep,
        title={Deep long-tailed learning: A survey},
        author={Zhang, Yifan and Kang, Bingyi and Hooi, Bryan and Yan, Shuicheng and Feng, Jiashi},
        journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
        year={2023},
        publisher={IEEE}
  }
  ```

  ## 5. Other Resources

  - [Papers With Code: Long-tailed Learning](https://paperswithcode.com/task/long-tail-learning)
  - [zzw-zwzhang/Awesome-of-Long-Tailed-Recognition](https://github.com/zzw-zwzhang/Awesome-of-Long-Tailed-Recognition)
  - [SADE/Test-Agnostic Long-Tailed Recognition](https://github.com/Vanint/SADE-AgnosticLT)

  
