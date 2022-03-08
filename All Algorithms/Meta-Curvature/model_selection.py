# Import all of the model classes and generation functions
#from models.AST import AST
from models.ResNet import ResNetModel
#from models.FSD50_CRNN import FSD50_CRNN
from models.Conv_and_Hybrid import StandardCNN, StandardHybrid


###############################################################################
# HELPER FUNCITONS
###############################################################################
def grab_model(name, mod, out_dim):
    if name == 'CNN':
        model = StandardCNN(out_dim=out_dim, **mod)
    elif name == 'Hybrid':
        model = StandardHybrid(out_dim=out_dim, **mod)
    elif name == 'ResNet18':
        model = ResNetModel(num_classes=out_dim, **mod)
    elif name == 'ResNet34':
        model = ResNetModel(num_classes=out_dim, **mod)

    elif name == 'FSD50K_CRNN':
        model = FSD50_CRNN(out_dim=out_dim, **mod)

        """
        elif name == 'AFT_full':
            model = AFT(out_dim=out_dim, **mod)
        elif name == 'AFT_simple':
            model = AFT(out_dim=out_dim, **mod)
        elif name == 'AFT_local':
            model = AFT(out_dim=out_dim, **mod)
        """

    elif name == 'AST_tiny':
        model = AST(out_dim=out_dim, **mod)
    elif name == 'AST_small':
        model = AST(out_dim=out_dim, **mod)
    elif name == 'AST_base':
        model = AST(out_dim=out_dim, **mod)
    elif name == 'ResNet12':
        model = ResNet12(out_dim=out_dim, **mod)
    
    return model