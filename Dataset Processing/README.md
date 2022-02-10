# Processing Pipelines

## Basic Description
This folder and its sub-parts make up the preprocessing pielines used for the 5 datasets covered in this work. The intent behind these scripts was to perform as much processing of the datasets offline as possible. In general, the offline processing of a dataset looks something like:
  - Obtain the dataset from source (sources for teh sets used in this work are given in this .md file)
  - Sort any already partitioned or non-structured data into a more standardised strucure which looks like one folder containing folders of all unique classes, where each class folder contains every sample which has its label attatched to it (this step does not apply to all datasets however does apply to a few which have propietary meta-data and meta-data structure to parse through)
  - Once this base structure is reached, this folder of class folders is mirrored where each sample is now saved as a .npy instead of a .wav/some other audo format (specifically for BirdClef, data samples from source had to be converted to more suitable forms whcih cost significant storage). Each sample has also been z-normalised across its length (this may not be suitable if wanting to use the dataset samples as their original time-series signal, however is for our use case of spectrograms)  
  - The mirror directory is then mirrored a second time (could edit codes to replace if storage is an issue), where each sample is now converted into a log-mel spectrogram and stored again as an .npy file

## General Scripts
Some of the loose files contained in this directory are general purpose and can be used for multiple/all of the datasets, these contain:
  - to_np_and_norm.py (converting raw .wav files to a npt format for faster loading into other files and scripts)
  - to_spec.py (converts the .npy raw audio files into log-mel spectrograms and re-stores them in .npy)
  - to_var_spec.py (converts variable length .npy samples into fixed length spectrogram representations)

## Combo Scripts
In addition to these files as well as the more specific ones included in the sub-directory are some example full stack processing pielines for a variety of the datasets. These can be edited and modified to suit specific needs but should help illustrate how all the processing scripts fit together. These are namely:
  - full_stack_ESC.py
  - full_stack_NSYNTH.py

## Other Considerations
For working with some of these datasets, we perform some cleaning/clipping in order to make the training and evaluation more tractable on consumer grade GPUs. This primarily involves the removal of incredibly long and rare clips. This filtering is only done on two occasions, these are:
  - For the Watkins mammal dataset, where there are 1-3 clips ~20 minutes long present in the dataset. We remove all samples longer than 3 minutes for this set which results in 8 removed clips in total.
  - For the whole dataset pruning of BirdClef as detailed further below

## BirdClef Pruning
The BirdClef dataset in its main and raw form is incredibly variable in length, spanning samples of 3 seconds to 30 minutes. Although samples longer than a few minutes are in the minority of the set, their inclusion requires larger memory GPUs to be used with a huge amount of headroom assumed. This directly goes against our goal with reproducibility and so we define a pruned version of the set which is much more internally consistent. Specifically we do two things, we remove:
  - Samples longer than 3 minutes (a max value chosen based off of the well-behaving VoxCeleb dataset)
  - Classes with less than 50 samples present after long samples have already been eradiacted
 
We perform experiments mainly with the pruned version of BirdClef however also include a base meta-learner table of results for the full untrimmed version.

## Dataset Sources
Sources for datasets:
  - https://github.com/karolpiczak/ESC-50 (ESC-50)
  - https://magenta.tensorflow.org/datasets/nsynth (NSynth)
  - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html (VoxCeleb1)
  - https://zenodo.org/record/2552860#.Yd2sLGDP2Uk (FSDKaggle18)
  - https://www.aicrowd.com/clef_tasks/22/task_dataset_files?challenge_id=211 (BirdClef2020)
  - https://www.kaggle.com/ttahara/birdsong-resampled-train-audio-00 (An easier to get approx/variant of BirdClef2020)
  - Watkins Marine Mammal Sound Database:
    - https://cis.whoi.edu/science/B/whalesounds/index.cfm (Main source page)
    - https://archive.org/details/watkins_202104 (A zipped collected of all the data)
  - http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz (Speech Commands Dataset V2)
  - https://github.com/CHeggan/AudioSet-For-Meta-Learning (AudioSet Scraping)

