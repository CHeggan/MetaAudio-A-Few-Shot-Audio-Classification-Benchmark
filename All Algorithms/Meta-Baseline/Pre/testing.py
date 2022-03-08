from pickle import load
import torch


def load_model(model_path, base_model):
    base_model.load_state_dict(torch.load(model_path))
    return base_model

###############################################################################
#def save_without_head(model_path, base_model):


from models.Conv_and_Hybrid import StandardHybrid

mod = {'in_channels': 1, 'seq_layers': 1, 'seq_type':'RNN', 'bidirectional': False, 'hidden_channels':64, 'pool_dim':[3, 3]}

path = 'C:/Users/calum/OneDrive/PHD/2021/Code Base/Pre-training/Supervised/results/NSYNTH_pretrain_trainset/nsynth_pretrain_trainset_5_seed_0/best_val_model__04_08__15_57.pt'
model = StandardHybrid(out_dim=35, **mod)

model = load_model(path, model)

newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
print(newmodel)