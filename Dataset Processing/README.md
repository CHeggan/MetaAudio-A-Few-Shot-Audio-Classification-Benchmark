This folder and its sub-parts make up the preprocessing pielines used for the 5 datasets covered in this work. The intent behind these scripts was to perform as much processing of the datasets offline as possible. In general, the offline processing of a dataset looks something like:
  - Obtain the dataset from source (sources for teh sets used in this work are given in this .md file)
  - Sort any already partitioned or non-structured data into a more standardised strucure which looks like one folder containing folders of all unique classes, where each class folder contains every sample which has its label attacthed to it (this step does not apply to all datasets however does apply to a few which have propietary meta-data and meta-data structure to parse through)
  - Once this base structure is reached, this folder of class folders is mirrored where each sample is now saved as a .npy instead of a .wav/some other audo format (specifically for BirdClef, data samples from source had to be converted to more suitable forms whcih cost significant storage). Each sample has also been z-normalised across its length (this may not be suitable if wanting to use the dataset samples as their original time-series signal, however is for our use case of spectrograms)  
  - The mirror directory is then mirrored a second time (could edit codes to replace if storage is an issue), where each sample is now converted into a log-mel spectrogram and stored again as an .npy file


Some of the loose files contained in this directory are general purpose and can be used for multiple/all of the datasets, these contain:
  - to_np_and_norm.py (converting raw .wav files to a npt format for faster loading into other files and scripts)
  - to_spec.py (converts the .npy raw audio files into log-mel spectrograms and re-stores them in .npy)


In addition to these files as well as the more specific ones included in the sub-directory are some example full stack processing pielines for a variety of thedatasets. These can be edited and modified to suit specific needs but should help illustrate how all the processing scripts fit together. these are namely:
  - full_stack_ESC.py
  - full_stack_NSYNTH.py
  - full_stack_KAGGLE18.py


Sources for datasets:
  - https://github.com/karolpiczak/ESC-50 (ESC-50)
  - https://magenta.tensorflow.org/datasets/nsynth (NSynth)
  - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html (VoxCeleb1)
  - https://zenodo.org/record/2552860#.Yd2sLGDP2Uk (FSDKaggle18)
  - https://www.aicrowd.com/clef_tasks/22/task_dataset_files?challenge_id=211 (BirdClef2020)
  - https://www.kaggle.com/ttahara/birdsong-resampled-train-audio-00 (An easier to get approx/variant of BirdClef2020)
