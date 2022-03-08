"""
Contains the following transforms:
    -> Centering
    -> l2 normalisation
    -> centering and l2 normalisation
"""
##############################################################################
# IMPORTS
##############################################################################
from numpy import linalg as LA

##############################################################################
# TRANSFORMS
##############################################################################
def centering(base_mean, new_features):
    return new_features - base_mean

def l2_norm(x):
    x = x.to('cpu').detach()
    return x / LA.norm(x, ord=2, axis=1)[:, None]

def cl2n(base_mean, new_features):
    new_features = new_features.to('cpu').detach()
    temp = new_features - base_mean
    return l2_norm(temp)