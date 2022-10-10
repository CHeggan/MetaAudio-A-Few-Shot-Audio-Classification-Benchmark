## Dataset Splits

One of the main contributions of this work in a benchmark sense is the standardisation of dataset splits and processing pipelines. Throughout all of the experiments presented in our work, we employ some fixed and consistent split of any given dataset. We do this mainly as it allows for fair comparison between algorithmic performance on some dataset/setting, both in this work and in the future. 

This kind of standardisation is already present in few-shot imagery where set evaluation procedures exist for datasets like Mini-ImageNet and Omniglot. We aim to mimic this. 

## Loading
Splits can be loaded into a numpy array with the following command
```python
    class_splits = np.load(path_to_split, allow_pickle=True)
    train, val, test = class_splits
```

## Types of Split
In this work we experiment with many of the splits included in this repo, though not all, in an attempt to investigate some interesting and unique behaviors of few-shot acoustic classification. Included here, we have 3 different types of datasets splits:
  - Baseline (Randomly generated with no additional information taken into account)
  - Length shifted and stratified (Sample length distribution shifts or stratifications)
  - Meta-aware (Splits which are stratified over some meta-data)

Each of these types are explained further in their own respective section
