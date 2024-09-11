"""
Fully-connected residual network as a single deep learner.
Convert 2d to 3d pose.
From: https://github.com/Nicholasli1995/EvoSkeleton/blob/master/libs/model/model.py
"""

import torch.nn as nn
import torch


class ResidualBlock(nn.Module):
    """
    A residual block.
    """

    def __init__(self, linear_size, p_dropout=0.5, kaiming=True, leaky=False, activation=True):
        super(ResidualBlock, self).__init__()
        self.l_size = linear_size
        self.activation = activation
        if leaky:
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

        if kaiming:
            self.w1.weight.data = nn.init.kaiming_normal_(self.w1.weight.data)
            self.w2.weight.data = nn.init.kaiming_normal_(self.w2.weight.data)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        if self.activation:
            y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        if self.activation:
            y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


class FCModel(nn.Module):
    def __init__(self,
                 stage_id=1,
                 linear_size=1024,
                 num_blocks=2,
                 p_dropout=0.5,
                 norm_twoD=False,
                 kaiming=True,
                 refine_3d=False,
                 leaky=False,
                 dm=False,
                 input_size=32,
                 output_size=64,
                 activation=True,
                 use_multichar=False,
                 id_embedding_dim=16):
        """
        Fully-connected network.
        """
        super(FCModel, self).__init__()
        if use_multichar:
            self.embedding_layer = nn.Embedding(10, embedding_dim=id_embedding_dim)
            input_size += id_embedding_dim
        self.activation = activation
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_blocks = num_blocks
        self.stage_id = stage_id
        self.refine_3d = refine_3d
        self.leaky = leaky
        self.dm = dm
        self.input_size = input_size
        if self.stage_id > 1 and self.refine_3d:
            self.input_size += 16 * 3
            # 3d joints
        self.output_size = output_size

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.res_blocks = []
        for l in range(num_blocks):
            self.res_blocks.append(ResidualBlock(self.linear_size,
                                                 self.p_dropout,
                                                 leaky=self.leaky,
                                                 activation=activation))
        self.res_blocks = nn.ModuleList(self.res_blocks)

        # output
        
        
        self.w2 = nn.Linear(self.linear_size, self.output_size)
        if self.leaky:
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

        if kaiming:
            self.w1.weight.data = nn.init.kaiming_normal_(self.w1.weight.data)
            self.w2.weight.data = nn.init.kaiming_normal_(self.w2.weight.data)
        self.out_activation = nn.Sigmoid()
        self.use_multichar=use_multichar


    def forward(self, x, id_index=0):
        if self.use_multichar:
            input_feature = self.embedding_layer(id_index.long())
            x = torch.cat([input_feature, x], dim=1)
        y = self.get_representation(x)
        y = self.w2(y)
        y = self.out_activation(y)
        return y

    def get_representation(self, x):
        # get the latent representation of an input vector
        # first layer
        y = self.w1(x)
        y = self.batch_norm1(y)
        if self.activation:
            y = self.relu(y)
        y = self.dropout(y)

        # residual blocks
        for i in range(self.num_blocks):
            y = self.res_blocks[i](y)

        return y


def get_model(stage_id,
              refine_3d=False,
              norm_twoD=False,
              num_blocks=2,
              input_size=32,
              output_size=64,
              linear_size=1024,
              dropout=0.5,
              leaky=False,
              activation=True,
            use_multichar=False,
            id_embedding_dim=16
              ):
    model = FCModel(stage_id=stage_id,
                    refine_3d=refine_3d,
                    norm_twoD=norm_twoD,
                    num_blocks=num_blocks,
                    input_size=input_size,
                    output_size=output_size,
                    linear_size=linear_size,
                    p_dropout=dropout,
                    leaky=leaky,
                    activation=activation,
                    
                     use_multichar=use_multichar,
                     id_embedding_dim=id_embedding_dim
                    )
    return model


def prepare_optim(model, opt):
    """
    Prepare optimizer.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if opt.optim_type == 'adam':
        optimizer = torch.optim.Adam(params,
                                     lr=opt.lr,
                                     weight_decay=opt.weight_decay
                                     )
    elif opt.optim_type == 'sgd':
        optimizer = torch.optim.SGD(params,
                                    lr=opt.lr,
                                    momentum=opt.momentum,
                                    weight_decay=opt.weight_decay
                                    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=opt.milestones,
                                                     gamma=opt.gamma)
    return optimizer, scheduler


def get_cascade():
    """
    Get an empty cascade.
    """
    return nn.ModuleList([])

class FC(nn.Module):
    def __init__(self, input_size=32, output_size=64, kaiming=True):
        """
        Fully-connected network.
        """
        super(FC, self).__init__()

        self.w1 = nn.Linear(input_size, output_size)

        if kaiming:
            self.w1.weight.data = nn.init.kaiming_normal_(self.w1.weight.data)

    def forward(self, x):
        out = self.w1(x)
        return out

if __name__ == '__main__':
    import os
    
    # cascade = get_cascade()
    # for stage_id in range(2):
    #     cascade.append(get_model(stage_id + 1, refine_3d=False,
    #                              norm_twoD=False,
    #                              num_blocks=2,
    #                              input_size=16,
    #                              output_size=139,
    #                              linear_size=1024,
    #                              dropout=0.5,
    #                              leaky=False
    #                              ))
    # cascade.eval()
    n_rig = 61
    exp_dim = 16
    model_path_root = '/data/Workspace/Rig2Face/ckpt'
    model = get_model(1, refine_3d=False,
                                 norm_twoD=False,
                                 num_blocks=4, #2,
                                 input_size=n_rig,
                                 output_size=exp_dim,
                                 linear_size=1024, #1024,
                                 dropout=0.0,
                                 leaky=False
                                 )
    checkpoint = torch.load(os.path.join(model_path_root, 'model_model_20221129-130512.pt'))
    model.load_state_dict(checkpoint['state_dict'])
    print("load model {}".format(os.path.join(model_path_root,f'model_model_20221129-130512.pt')))
 
    rigs = torch.randn((8, n_rig))
    rigs = torch.clip(rigs, min=0, max=1)
    outputs = model(rigs)
    print(outputs)
    
