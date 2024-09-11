import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import model.models_vit as models_vit
 


class Pipeline(nn.Module):
    def __init__(self,config):
        super(Pipeline,self).__init__()
        self.config = config
        model_name = 'vit_base_patch16'
        num_classes = 16
        ckpt_path = config["mae_pretrain_checkpoints"]
        model = getattr(models_vit, model_name)(
        global_pool=True,
        num_classes=num_classes,
        drop_path_rate=0.1,
        img_size=224,
        )
        print(f"Load pre-trained checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        checkpoint_model = checkpoint['model']
        model.load_state_dict(checkpoint_model, strict=False)
        self.main = model

    def forward(self, x):
        x = self.main(x)
        x = F.normalize(x, dim=1)
        return x

    def forward_fea(self,x):
        x,fea = self.main(x,True)
        return fea
