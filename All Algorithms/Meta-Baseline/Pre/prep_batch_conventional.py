"""
Script hosts the much simpler conventional prep batch functions. Reason we 
    include here is that we want to be able to generalise to transformers which
    require some transposes in practice
"""


"""
This script is dedicated to the nested prep_batch function. The general idea
    here is that we initialise the function with parameters once and can then 
    pass it more basic data later on repititively. i.e the nested funtion gives
    back an intialised function we can use more easily and readily.
"""

###############################################################################
#IMPORTS
###############################################################################
import torch

###############################################################################
# NORMAL PREP BATCH FUNCTION
##############################################################################
def prep_batch(device):
    """Is the parent function for batch prepping. Returns an initialised function 

    Args:
        device (torch CUDA object): The CUDA device we want to load data to
    """
    def prep_batch(batch):
        """The child prep batch fucntion. Takes some batch and processes it
            into prper tasks before moving it to a GPU for calculations.

        Args:
            batch (Tensor): The unformatted batch of data and tasks

        Returns:
            Tensor, Tensor: The formatted x, y tensor pairs
        """
        x, y = batch

        y = y.long().to(device)
        x = x.unsqueeze(1).double().to(device)

        return x, y
    return prep_batch

###############################################################################
# TRANSFORMER PREP BATCH FUNCTION
##############################################################################
def prep_trans_batch(device):
    """Is the parent function for batch prepping. Returns an initialised function 

    Args:
        device (torch CUDA object): The CUDA device we want to load data to
    """
    def prep_trans_batch(batch, meta_batch_size):
        """The child prep batch fucntion. Takes some batch and processes it
            into prper tasks before moving it to a GPU for calculations.

        Args:
            batch (Tensor): The unformatted batch of data and tasks

        Returns:
            Tensor, Tensor: The formatted x, y tensor pairs
        """
        x, y = batch

        x = torch.transpose(x, 2, 3)

        y = y.long().to(device)
        x = x.double().to(device)

        return x, y
    return prep_trans_batch
