"""
Script exists to store the messier code for model selection for training 
"""
################################################################################
# IMPORTS
################################################################################
import torch

#from models.AST import AST
#from models.ResNet import ResNetModel
from models.Conv_and_Hybrid_ss import StandardHybrid, StandardCNN

################################################################################
# SELECTION
################################################################################
def grab_dual_model(name, mod, out_dim):
    if name == 'CNN':
        model = StandardCNN(out_dim=out_dim,  **mod)
    elif name == 'Hybrid':
        model = StandardHybrid(out_dim=out_dim, **mod)
    return model


