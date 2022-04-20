"""
Script exists to store the messier code for model selection for training 
"""
################################################################################
# IMPORTS
################################################################################
import torch

#from models.AST import AST
#from models.ResNet import ResNetModel
from models.Global_CNN_and_Hybrid_SS import GlobalCNN, GlobalHybrid

################################################################################
# SELECTION
################################################################################
def grab_dual_model(name, mod, out_dim):
    if name == 'CNN':
        model = GlobalCNN(out_dim=out_dim,  **mod)
    elif name == 'Hybrid':
        model = GlobalHybrid(out_dim=out_dim, **mod)
    return model


