# All Algorithms
## Included
For MetaAudio, we primarily experiment with 5 algorithms. These are the following:
  -  [MAML](https://arxiv.org/abs/1703.03400)
  -  [Meta-Curvature](https://arxiv.org/abs/1902.03356)
  -  [Prototypical Networks](https://arxiv.org/abs/1703.05175)
  -  [SimpleShot](https://arxiv.org/abs/1911.04623)
  -  [Meta-Baseline](https://arxiv.org/abs/2003.04390)

We use all of these algorithms multiple times over in different training and evaluation strategies. Providing every version and subset of each algorithm code would only serve to clutter this repo and so we instead opt to provide the primary code bases for each of the algorithms (excluding the [examples](https://github.com/CHeggan/MetaAudio-A-Few-Shot-Audio-Classification-Benchmark/tree/main/Examples)), including additional example scripts that serve as (almost) drop-in replacements. 

## The Code
The code bases we do provide function for both fixed and variable length spectrogram datasets, however not for raw form audio (however the changes needed for this could be later added upon request). Each of these code sets are neatly packaged so they can be largely used on their own. This does result in the addition of duplicate scripts (especially for dataset loading etc) however we feel the extra ease of experiment set up is worth the minor extra space the whole repo carries. 