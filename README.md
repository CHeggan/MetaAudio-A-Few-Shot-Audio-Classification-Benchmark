# MetaAudio-A-Few-Shot-Audio-Classification-Benchmark
## Citation
A new comprehensive and diverse few-shot acoustic classification benchmark. If you use any code or results from results from this work, please cite the following: 
```
@article{MetaAudio,
  author = {Calum Heggan et al.},
  title = {Meta-Audio: A Few-Shot Audio Classification Benchmark},
  year = {2022},
  publisher = {ICANN},
}
```

## Enviroment


## Contents Overview
This repo contains the following:
 - Multiple problem statement setups with accompanying results which can be used moving forward as baselines for few-shot acoustic classification. These include:
   - Normal within-dataset generalisation 
   - Joint training to both within and cross-dataset settings
   - Additional data -> simple classifier for cross-dataset
   - Length shifted and stratified problems for variable length dataset setting
 - Standardised meta-learning/few-shot splits for 5 distinct datasets from a variety of sound domains. This includes both baseline (randomly generated splits) as well as some more unique and purposeful ones such as those based on available meta-data and sample length distirbuions
 - Variety of algorithm implementations designed to deal with few-shot classification, ranging from 'cheap' traditional training pielines to SOTA Gradient-Based Meta-Learning (GBML) models
 - Both Fixed and Variable length dataset processing pielines

## Algorithm Implementations
Algorithms are custom built, operating on a similar framework with a common set of scripts. Those included are as follows:
  -  [MAML](https://arxiv.org/abs/1703.03400)
  -  [Meta-Curvature](https://arxiv.org/abs/1902.03356)
  -  [Prototypical Networks](https://arxiv.org/abs/1703.05175)
  -  [SimpleShot](https://arxiv.org/abs/1911.04623)
  -  [Meta-Baseline](https://arxiv.org/abs/2003.04390)

For both MAML & Meta-Curvature we also make use of the [Learn2Learn](https://arxiv.org/abs/2008.12284) framework.

## Datasets
We primarily cover 5 datasets for the majority of our experimentation, these are as follows:
  - [ESC-50](https://www.karolpiczak.com/papers/Piczak2015-ESC-Dataset.pdf)
  - [NSynth](https://arxiv.org/abs/1704.01279)
  - [FSDKaggle18](https://arxiv.org/abs/1807.09902)
  - [BirdClef2020](https://www.imageclef.org/BirdCLEF2020)

In addition to these however, we also include 2 extra datasets for cross-dataset testing:
  - [Watkins Marine Mammal Sound Database](https://cis.whoi.edu/science/B/whalesounds/index.cfm)
  - [SpeechCommandsV2](https://arxiv.org/abs/1804.03209)

as well as a propietary version of AudioSet we use for pre-training with simple classifiers. We obtained/scraped this dataset using the code from [here](https://github.com/CHeggan/AudioSet-For-Meta-Learning):
  - [AudioSet](https://ieeexplore.ieee.org/abstract/document/7952261)

We include sources for all of these datasets in [Datset Processing](https://github.com/CHeggan/MetaAudio-A-Few-Shot-Audio-Classification-Benchmark/tree/main/Dataset%20Processing)





