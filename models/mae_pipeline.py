import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import models.models_vit as models_vit


class Emb_mae(nn.Module):
    def __init__(self):
        super(Emb_mae,self).__init__()

        state_dict = torch.load("/data/Workspace/Rig2Face/ckpt/emb_regress_loss_0.0088.pth")
        new_dict = {}
        for k,v in state_dict.items():
            if "module." in k:
                new_k = k.split("module.")[1]
                new_dict[new_k] = v
        if len(new_dict)>0:
                state_dict = new_dict
            
        emb_net = Pipeline_mae().cuda()
        emb_net.load_state_dict(state_dict)
        emb_net.eval()
        for params in emb_net.parameters():
            params.requires_grad = False
        self.emb_net = emb_net
        # self.transform = build_transform(False)
    
    def forward(self, img_tensor):
        # feature_mae = self.transform(img_tensor)
        emb = self.emb_net(img_tensor)
        return emb, emb

class Pipeline_mae(nn.Module):
    def __init__(self):
        super(Pipeline_mae,self).__init__()
        model_name = 'vit_base_patch16'
        num_classes = 16
        model = getattr(models_vit, model_name)(
        global_pool=True,
        num_classes=num_classes,
        drop_path_rate=0.1,
        img_size=224,
        )
        self.main = model

    def forward(self,x):
        x,fea = self.main(x,True)
        x = F.normalize(x, dim=1)
        return fea,x
        
