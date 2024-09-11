import torch
import torchvision.transforms.transforms as transforms
from models.mae_pipeline import Pipeline_mae
from models.Emoca_ExprNet import ExpressionLossNet as EmoNet

def load_emb_model(backbone, opt=None):
      
    if backbone == 'emonet':
        model_emb = EmoNet().cuda()
        resize = transforms.Compose([transforms.Resize([224,224], antialias=True)])
        n_rig = 2048
        if opt:
            opt.exp_dim = 2048

    elif backbone == 'mae_emb':
        n_rig = 768
        mean = [0.49895147219604985, 0.4104390648367995, 0.3656147590417074]
        std = [0.2970847084907291, 0.2699003075660314, 0.2652599579468044]
        resize = transforms.Compose([transforms.Resize([224,224], antialias=True),
                                     transforms.Normalize(mean, std)]) 
        model_emb = Pipeline_mae()
        ckpt_mae = torch.load('/data/Workspace/Rig2Face/ckpt/epoch_90_acc_0.8736.pth')
        ckpt_mae = {key.replace('module.', ''):ckpt_mae[key] for key in ckpt_mae.keys()}
        model_emb.load_state_dict(ckpt_mae)    
        if opt:
            opt.exp_dim = n_rig

    return model_emb, n_rig, resize