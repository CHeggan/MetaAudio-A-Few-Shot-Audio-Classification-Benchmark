# Examples & Walkthrough
Included here are some full walkthroughs for testing the repo and its contained experiments. We include two different full examples:
  - MAML for ESC-50
  - ProtoNets for Kaggle18

These experiments assume that you have setup an anaconda/miniconda environment with the required packages installed. Steps for this can be found on the main repo page.

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
  - Run the full_stack_ESC.py file with the relevant conda environment activated. 3 new data folders should be created:
    -  Audio data sorted in its .wav format
    -  Audio data in z-normalised .npy files 
    -  Audio spectrograms (this is the folder we want to get a path to)

Now that the data has been processed, we take note of the exact path to the folder containing the audio spectrograms, e.g. 'X:/Datasets/ESC-50-master/ESC_spec'

### Step 2: MAML Setup & Experiment
  - Navigate to the 'maml_experiment_params.yaml' file in the 'MAML_ESC' code folder
  - Change the 'task_type' variable to whatever you like (assumes string). This is the name of the experiment that will be attached with final results and detailed data on training and evaluation
  - Change the 'data_path' variable to the spectrogram folder path collected from the previous step
  - execute the following in your command line with the relevant anaconda environment enabled:
    - > python BaseLooper.py

## ProtoNets for Kaggle18
### Step 1: Getting & Formatting Data
  - Download all files from here: https://zenodo.org/record/2552860#.Yd2sLGDP2Uk (there should be 4 main folders)
  - Unzip all of the downloaded files and place them into a master folder (e.g. /Kaggle18_Set_Master)
  - Place the following files in the master folder:
    - to_var_spec.py
    - to_np_and_norm.py
    - folder_sort_KAGGLE18.py (can be found in the 'Specific Scripts' folder)
    - full_stack_KAGGLE.py
  - In the full_stack_KAGGLE.py file, change the MAIN_DIR variable to the exact path to the the master data folder containing the unzipped folders, see its current value in the script for an example. Also set the SAMPLE_LENGTH variable to the fixed length representation value you want to use, we suggest 5 (see paper for details). 
  - Run the full_stack_KAGGLE.py file with the relevant conda environment activated. 3 new data folders should be created:
    -  Audio data sorted in its .wav format
    -  Audio data in z-normalised .npy files 
    -  Stacked audio spectrograms (this is the folder we want to get a path to)

### Step 2: ProtoNet Setup & Experiment
  - Navigate to the 'proto_params.yaml' file in the 'Proto_Kaggle18' code folder
  - Change the 'task_type' variable to whatever you like (assumes string). This is the name of the experiment that will be attached with final results and detailed data on training and evaluation
  - Change the 'data_path' variable to the spectrogram folder path collected from the previous step
  - execute the following in your command line with the relevant anaconda environment enabled:
    - > python BaseLooperProto.py

## Extraction Of Results
Final results along with 95% confidence intervals should be automatically output to a .txt file in the main algorithm folder. For additional details, can navigate to the results folder to find the experiment name setup (the seed randomly chosen and used is automatically attached to the name of the folder for reference). In this experiment specific sub-folder you can find the following:
  - The best model (.pt) file based on validation set
  - Some image file to summarise training/evaluation statistics
  - Training and validation details (.csv files)
