Here we have the length distribution shifted and stratified dataset splits. 

The idea behind defining these splits was to investigate how either similar or mismatched sample length distrbutions between train and test times impacted performance of our models. For each variable length dataset (Kaggle18, VoxCeleb1, BirdClef2020) we define both a shifted and stratified split:
  - Shifted: Making difference between train and test distributions larger
  - Stratified: Making difference smaller
  
Actually creating these splits can easily become a fairly complex optimisation problem itself, however to avoid this and the time/computational costs associated wetake a saimple and intuitive approach. This is outlined more in the paper however the basic methodology is as follows:
  - Calculate an expected average sample length for each class based on some % of its total samples
  - Sort these expected averages in ascending or descending order
  - Two different options for shift/strat:
  -   For shift, take first n classes in the sorted list to reach desired training ratio of full set of classes, repeat with leftover classes for validation and then finally testing. Taking training first and leaving testing to last should create a natural distribution shift with respect to sample lengths
  -   For each split element in strat (train/val/test), select classes from both sides of the sorted list each time a selection is made and repeat until desired ratios are met. The intuition here is that the expected shortest length class is in the same split as the longest, and so on, creating a widened distribution for all splits
