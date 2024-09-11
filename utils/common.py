import imageio
import yaml
import torch
import random
from torch import nn
import pickle
import torch.nn.init as init
import numpy as np
from types import SimpleNamespace
import os

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    # 为其他类型的层添加初始化，例如卷积层
    elif isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:  # 对于权重矩阵
                init.xavier_uniform_(param.data)
            else:  # 对于偏置项
                init.zeros_(param.data)
                
def tensors_to_cuda(data):
    """
    Recursively move all tensors in the input data to CUDA, if CUDA is available.
    
    :param data: A dictionary which may contain other dictionaries or torch.Tensors.
    :return: Same structure as input with all tensors moved to CUDA.
    """
    if torch.cuda.is_available():
        if isinstance(data, dict):
            # Recursively apply to dictionary elements
            return {key: tensors_to_cuda(value) for key, value in data.items()}
        elif isinstance(data, torch.Tensor):
            # Move tensor to CUDA
            return data.to('cuda')
        else:
            # If data is not a dictionary or tensor, return it as is
            return data
    else:
        # If CUDA is not available, return the data unchanged
        return data
    
    
def parse_args_from_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    config = SimpleNamespace(**config)

    return config

def imgs2video(image_folder, video_name='', fps=30):
    if not video_name:
        video_name = image_folder+'.mp4'

    images = [img for img in os.listdir(image_folder) if img.split('.')[-1] in ['png', 'jpg', 'jpeg']]
    images.sort()

    frames = []
    for image_file in images:
        frames.append(imageio.imread(os.path.join(image_folder, image_file)))
    imageio.mimsave(video_name, frames, 'FFMPEG', fps=fps)
    
    print('output video: ', video_name)
    return True

def count_parameters(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        for name, param in model.named_parameters():
            print(f'Layer: {name} | Parameters: {param.numel()}')
    return {"Total": total_params, "Trainable": trainable_params, "Non-Trainable": total_params - trainable_params}