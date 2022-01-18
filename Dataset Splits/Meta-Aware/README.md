## Meta-Aware Splits

For both of our fixed length datasets (ESC-50 & NSynth) we define some meta-data aware stratified splits, motivated by the possibility for further community experimentation. For both of these sets we look at using major class groupings in order to do this, where for each major grouping we share their classes equally (by weight, i.e 70% for training) between train/val/test. Intuitively, this should make the meta-learning problem statements for these datasets easier as each major catagory is represented in all three of components of the dataset split.

# ESC-50
ESC-50 can be loosely broken down into 5 major categories including Animals, Natural soundscapes & water sounds, Human non-speech, Interior/domestic sounds and Exterior/urban noises. These class groupings were not originally included in any meta-data file from source (is on the main repo in a table) and so we also include a simple relation table here (major_groupings.csv). 

# NSynth
From source, NSynth's meta-data is sufficiently detailed to create a variety of interesting meta splits. We choose to consider the instrument component to stratify over, however leave open the possibility for other aspects to be considered at a later date. In total there are 11 instruments which we equally share amongst train/val/test.
