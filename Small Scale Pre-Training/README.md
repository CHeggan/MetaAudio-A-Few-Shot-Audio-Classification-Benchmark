# Small-Scale Pre-Training
In our external data track of MetaAudio, we experiment with the idea of pre-training a general purpose encoder, which we then use with a simple linear classifier to solve few-shot tasks at test time. Our original set of experimnets (not included in the main paper) focused on using some small porpietary subset of AudioSet (~30k samples), downloaded and refined using the tools in this [repo](https://github.com/CHeggan/AudioSet-For-Meta-Learning). This section outlines this set of experiments. 

## Whats Included?
Due to licensing we can not release the data we scraped and used from the ontology directly. However due to the nature of the dataset, it is likely that many samples we used are no longer available. To get around best as possible we include the fully pre-trained model (our GlobalCRNN) using our specific subset of AudioSet. Specifically we include 2 files:
 - best_val_model.pt (The GlobalCRNN model pre-trained with AudioSet subset)
 - best_features.npy (A file of best feature extractions for each data sample from the model after it has been fully trained)

To be clear, we don't expect much work to continue with this specific model, and encourage future researchers to build upon this external data track in a more fully reproducible way.  For this reason we also include the full code we used for pre-training the CRNN model. 


