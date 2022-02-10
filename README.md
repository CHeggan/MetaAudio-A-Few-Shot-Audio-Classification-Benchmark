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

## Algorithms & Datasets Implementations
Algorithms are all custom built, operating on a similar framweork with a common set of scripts. Those included are as follows:
  -  MAML [[1]](https://arxiv.org/abs/1703.03400)
  -  Meta-Curvature [[2]](https://arxiv.org/abs/1902.03356)
  -  Prototypical Networks [[3]](https://arxiv.org/abs/1703.05175)
  -  SimpleShot [[4]](https://arxiv.org/abs/1911.04623)
  -  Meta-Baseline [[5]](https://arxiv.org/abs/2003.04390)

## Dataset Sources
Sources for datasets:
  - https://github.com/karolpiczak/ESC-50 (ESC-50)
  - https://magenta.tensorflow.org/datasets/nsynth (NSynth)
  - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html (VoxCeleb1)
  - https://zenodo.org/record/2552860#.Yd2sLGDP2Uk (FSDKaggle18)
  - https://www.aicrowd.com/clef_tasks/22/task_dataset_files?challenge_id=211 (BirdClef2020)
  - https://www.kaggle.com/ttahara/birdsong-resampled-train-audio-00 (An easier to get approx/variant of BirdClef2020)



