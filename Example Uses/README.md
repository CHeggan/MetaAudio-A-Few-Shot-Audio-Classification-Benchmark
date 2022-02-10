# Examples & Walkthrough
Included here are some full walkthorughs for testing the repo and its contained experiments. We include two different full examples:
  - MAML for ESC-50
  - ProtoNets for Kaggle18

These expeirments assume that you have setup an anaconda/miniconda enviroment with the required packages installed. Steps for this can be found on the main repo page

## MAML for ESC-50
### Step 1: Getting & Formatting Data
  - Dowload the full master file for the ESC-50 dataset from https://github.com/karolpiczak/ESC-50 (should be a tag saying 'download .zip')
  - Unzip the dataset and navigate inside of the new folder
  - Place the following files in 'ESC-50-master':
    - to_spec.py
    - to_np_and_norm.py
    - folder_sort_ESC.py (can be found in the 'Specific Scripts' folder)
    - full_stack_ESC.py
  - In the full_stack_ESC.py file, change the MAIN_DIR variable to the exact path to the the 'ESC-50-master' folder, see its current value in the script for an example 
  - Run the full_stack_ESC.py file. 3 new data folder should be created:
    -  Audio data sorted in its .wav format
    -  Audio data in z-normalised .npy files 
    -  Audio spectrograms (this is the folder we want to get a path to)

Now that the data has been processed, we take note of the exact path to the folder containing the audio spectrograms, e.g. 'X:/Datasets/ESC-50-master/ESC_spec'

### Step 2: Algorithm Setup & Experiment
  - Navigate to the 'maml_experiment_params.yaml' file in the 'MAML_ESC' code folder
  - Change the 'data_path' variable to the spectrogram folder path collected from the previous step
  - execute the following < python BaseLooper.py



## ProtoNets for Kaggle18
### Step 1: Getting & Formatting Data
Download all files from here: https://zenodo.org/record/2552860#.Yd2sLGDP2Uk (there should be 4 main folders)
