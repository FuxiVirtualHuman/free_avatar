# -*- coding: utf-8 -*-
# @Author: aaronlai
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    from datasets.process_ctls_emb import load_ctr_from_txt
    from datasets.RigData import read_image,get_statis
except:
    pass
'''
DCGAN From: https://github.com/AaronYALai/Generative_Adversarial_Networks_PyTorch
'''

class DCGAN_Discriminator(nn.Module):
    def __init__(self, featmap_dim=512, n_channel=1):
        super(DCGAN_Discriminator, self).__init__()
        self.featmap_dim = featmap_dim
        self.conv1 = nn.Conv2d(n_channel, int(featmap_dim / 4), 5,
                               stride=4, padding=2)

        self.conv2 = nn.Conv2d(int(featmap_dim / 4), int(featmap_dim / 2), 5,
                               stride=2, padding=2)
        self.BN2 = nn.BatchNorm2d(int(featmap_dim / 2))

        self.conv3 = nn.Conv2d(int(featmap_dim / 2), featmap_dim, 5,
                               stride=2, padding=2)
        self.BN3 = nn.BatchNorm2d(featmap_dim)

        self.conv4 = nn.Conv2d(featmap_dim, featmap_dim, 5,
                               stride=2, padding=2)
        self.BN4 = nn.BatchNorm2d(featmap_dim)

        self.conv5 = nn.Conv2d(featmap_dim, featmap_dim, 5,
                               stride=2, padding=2)
        self.BN5 = nn.BatchNorm2d(featmap_dim)

        self.fc = nn.Linear(featmap_dim * 4 * 4, 1)

    def forward(self, x):
        """
        Strided convulation layers,
        Batch Normalization after convulation but not at input layer,
        LeakyReLU activation function with slope 0.2.
        """
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.BN2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.BN3(self.conv3(x)), negative_slope=0.2)
        x = F.leaky_relu(self.BN4(self.conv4(x)), negative_slope=0.2)
        x = F.leaky_relu(self.BN5(self.conv5(x)), negative_slope=0.2)
        x = x.view(-1, self.featmap_dim * 4 * 4)
        x = F.sigmoid(self.fc(x))
        return x


class DCGAN_Generator(nn.Module):

    def __init__(self, featmap_dim=1024, n_channel=1, noise_dim=100):
        super(DCGAN_Generator, self).__init__()
        self.featmap_dim = featmap_dim
        self.fc1 = nn.Linear(noise_dim, 4 * 4 * featmap_dim)
        self.conv1 = nn.ConvTranspose2d(featmap_dim, (featmap_dim / 2), 5,
                                        stride=2, padding=2)

        self.BN1 = nn.BatchNorm2d(featmap_dim / 2)
        self.conv2 = nn.ConvTranspose2d(featmap_dim / 2, featmap_dim / 4, 6,
                                        stride=2, padding=2)

        self.BN2 = nn.BatchNorm2d(featmap_dim / 4)
        self.conv3 = nn.ConvTranspose2d(featmap_dim / 4, n_channel, 6,
                                        stride=2, padding=2)

    def forward(self, x):
        """
        Project noise to featureMap * width * height,
        Batch Normalization after convulation but not at output layer,
        ReLU activation function.
        """
        x = self.fc1(x)
        x = x.view(-1, self.featmap_dim, 4, 4)
        x = F.relu(self.BN1(self.conv1(x)))
        x = F.relu(self.BN2(self.conv2(x)))
        x = F.tanh(self.conv3(x))

        return x

# Define the Generator Network
class Generator(nn.Module):
    '''
    From https://github.com/Natsu6767/DCGAN-PyTorch/blob/master/dcgan.py
    '''
    def __init__(self, params, activation='sigmoid', convert_norm=False):
        super().__init__()

        # Input is the latent vector Z.
        # self.fc1 = nn.Linear(139, 64*8)
        # self.fc2 = nn.Linear(64*8, 64*4)
        # TODO params['nz'] / 64*4
        self.params = params
        self.activation = activation
        self.tconv1 = nn.ConvTranspose2d(params['nz'], params['ngf']*8,
            kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(params['ngf']*8)
        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(params['ngf']*8, params['ngf']*4,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ngf']*4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(params['ngf']*4, params['ngf']*2,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ngf']*2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(params['ngf']*2, params['ngf'],
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ngf'])

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(params['ngf'], params['ngf'],
            4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(params['ngf'])


        self.tconv6 = nn.ConvTranspose2d(params['ngf'], params['ngf'],
                                         4, 2, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(params['ngf'])

        # self.tconv7 = nn.ConvTranspose2d(params['ngf'], params['ngf'],
        #                                  4, 2, 1, bias=False)
        # self.bn7 = nn.BatchNorm2d(params['ngf'])


        self.tconv8 = nn.ConvTranspose2d(params['ngf'], params['nc'],
                                         4, 2, 1, bias=False)
        self.upsample = nn.Upsample(scale_factor=2)
        #Output Dimension: (nc) x 64 x 64
        # self.x_neu, self.out_nue = self.get_neutral()
        # self.mean, self.std = np.load('./results/data_statis.npy')
        # self.mean = torch.tensor(self.mean).cuda().reshape(1,-1,1,1).float()
        # self.std = torch.tensor(self.std).cuda().reshape(1,-1,1,1).float()

        # self.convert_norm = convert_norm

    def get_neutral(self):
        img_path = '/project/ard/3DFacialExpression/Images/ZHEN_v3_all/Images/sample_3591_.0000.jpg'
        rig_path = '/project/ard/3DFacialExpression/Ctrls/ZHEN/sample_3591_CtrlRigs.txt'
        mean, std = get_statis()
        img = read_image(img_path, mode='rgb', size=256)
        rigs,_,_,_,_ = load_ctr_from_txt(rig_path, do_flip=False)
        img = np.array(img)/255.
        rigs = (rigs - mean) / std
        return torch.tensor(rigs).cuda().float().reshape(1,139,1,1), torch.tensor(img.transpose(2,0,1)).cuda().float().unsqueeze(0)

    def forward(self, x):
        # x = F.dropout(self.fc1(x.view(-1,139)))
        # x = F.dropout(self.fc2(x)).view(-1,64*4,1,1)
        # x = x - self.x_neu
        #
        # if self.convert_norm:
        #     x = ((x * 2 - 1) - self.mean)/self.std

        x1 = F.relu(self.bn1(self.tconv1(x)))
        x2 = F.relu(self.bn2(self.tconv2(x1)))
        x3 = F.relu(self.bn3(self.tconv3(x2)))
        x4 = F.relu(self.bn4(self.tconv4(x3)))
        x5 = F.relu(self.bn5(self.tconv5(x4)))
        out = F.relu(self.bn6(self.tconv6(x5)))
        # x6 = self.upsample(x5) + x6
        # x = F.relu(self.bn7(self.tconv7(x)))
        if self.params['nc'] !=3:
            out = x4

        if self.activation == 'tanh':
            x7 = F.tanh(self.tconv8(out))
        elif self.activation =='sigmoid':
            x7 = F.sigmoid(self.tconv8(out))
        else:
            raise NotImplementedError
        # x7 = F.sigmoid(self.tconv8(x6) - self.out_nue)
        return x7


class Generator_rectangle(nn.Module):
    '''
    From https://github.com/Natsu6767/DCGAN-PyTorch/blob/master/dcgan.py
    '''
    def __init__(self, params, activation='sigmoid', convert_norm=False):
        super().__init__()

        # Input is the latent vector Z.
        # self.fc1 = nn.Linear(139, 64*8)
        # self.fc2 = nn.Linear(64*8, 64*4)
        # TODO params['nz'] / 64*4
        self.params = params
        self.activation = activation
        self.tconv1 = nn.ConvTranspose2d(params['nz'], params['ngf']*8,
            kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(params['ngf']*8)
        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(params['ngf']*8, params['ngf']*4,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ngf']*4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(params['ngf']*4, params['ngf']*2,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ngf']*2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(params['ngf']*2, params['ngf'],
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ngf'])

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(params['ngf'], params['ngf'],
            4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(params['ngf'])


        self.tconv8 = nn.ConvTranspose2d(params['ngf'], params['nc'],
                                         4, 2, 1, bias=False)
        self.pooling = nn.MaxPool2d((4,1),(4,1))
        
    def forward(self, x):
        x1 = F.relu(self.bn1(self.tconv1(x)))
        x2 = F.relu(self.bn2(self.tconv2(x1)))
        x3 = F.relu(self.bn3(self.tconv3(x2)))
        x4 = F.relu(self.bn4(self.tconv4(x3)))
        x5 = F.relu(self.bn5(self.tconv5(x4)))
        
        if self.params['nc'] !=3:
            out = x4

        out = self.pooling(x5)
        if self.activation == 'tanh':
            x7 = F.tanh(self.tconv8(out))
        elif self.activation =='sigmoid':
            x7 = F.sigmoid(self.tconv8(out))
        else:
            raise NotImplementedError
        return x7


if __name__ == '__main__':
    params = {'nz':4, 'ngf':64, 'nc':3}
    model = Generator_rectangle(params)
    inputs = torch.randn((8, 4,1,1))
    outputs = model(inputs)

    # inputs = torch.randn((8, 3,256,256))
    # model = DCGAN_Discriminator(n_channel=3)
    # out = model(inputs)
    # print(1)