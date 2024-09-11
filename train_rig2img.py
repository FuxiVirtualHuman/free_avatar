from tqdm import tqdm

def Train(epoch, loader, model):
    lr = optimizer.param_groups[0]['lr']
    print(f"*** Epoch {epoch}, lr:{lr:.5f}, timestamp:{timestamp}")
    loss_sum = 0.0
    model.train()    
    train_step = min(args.train_step_per_epoch, len(loader))
    b = '{l_bar}{bar:20}{r_bar}{bar:10b}'
    pbar = tqdm(enumerate(loader), bar_format=b, total=train_step)
    logger = []
    time0 = time.time()
    for i, data in pbar:
        if i>train_step:
            break
        optimizer.zero_grad()
        loss = dict()

        targets = data['img'].cuda().float()
        rigs = data['rigs'].cuda().float()
        assert (data['has_rig'] == 1).all()
        outputs = model(rigs.reshape(-1, configs_character['n_rig'], 1, 1))
        loss['image'] = criterion_l1(outputs, targets) * args.weight_img
        loss['mouth'] = criterion_l1(outputs*mouth_crop, targets*mouth_crop) * args.weight_mouth

        loss_value = sum([v for k, v in loss.items()])
        
        loss_sum += loss_value.item()
        loss_value.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        

        writer.add_scalars(f'train/loss', loss, epoch * train_step + i)
        writer.add_scalar(f'train/loss_total', loss_value.item(), epoch * train_step + i)
        
        _loss_str = str({k: "{:.4f}".format(v/(i+1)) for k, v in loss.items()})
        _log = f"Epoch {epoch}({timestamp}) (lr:{optimizer.param_groups[0]['lr']:05f}): [{i}/{len(train_dataloader)}] loss_G:{_loss_str}"
        logger.append(_log+'\n')
        pbar.set_description(_log)
        
    writer.add_images(f'train/img', torch.cat([outputs, targets], dim=-2)[::4], epoch * train_step + i)
    avg_loss = loss_sum / train_step
    _log = "==> [Train] Epoch {} ({}), training loss={}".format(epoch, timestamp, avg_loss)
    print(_log)
    with open(os.path.join(log_save_path, f'{task}_{timestamp}.log'), "a+") as log_file:
        log_file.writelines(logger)
    if epoch % args.save_step == 0:
        torch.save({'state_dict': model.state_dict()}, model_path.replace('.pt', f'_{epoch}.pt'))
    return avg_loss

def Eval(epoch, loader, model, best_score):
    loss_sum = 0.0
    model.eval()    
    eval_step = min(args.eval_step_per_epoch, len(loader))
    b = '{l_bar}{bar:20}{r_bar}{bar:10b}'
    pbar = tqdm(enumerate(loader), bar_format=b, total=eval_step)
    logger = []
    time0 = time.time()
    for i, data in pbar:
        if i>eval_step:
            break
        loss = dict()

        targets = data['img'].cuda().float()
        rigs = data['rigs'].cuda().float()
        assert (data['has_rig'] == 1).all()
        with torch.no_grad():
            outputs = model(rigs.reshape(-1, configs_character['n_rig'], 1, 1))
            loss['image'] = criterion_l1(outputs, targets) * args.weight_img
            loss['mouth'] = criterion_l1(outputs*mouth_crop, targets*mouth_crop) * args.weight_mouth

        loss_value = sum([v for k, v in loss.items()])
        
        loss_sum += loss_value.item()

        writer.add_scalars(f'train/loss', loss, epoch * eval_step + i)
        writer.add_scalar(f'train/loss_total', loss_value.item(), epoch * eval_step + i)
        
        _loss_str = str({k: "{:.4f}".format(v/(i+1)) for k, v in loss.items()})
        _log = f"Epoch {epoch}({timestamp}) (lr:{optimizer.param_groups[0]['lr']:05f}): [{i}/{len(train_dataloader)}] loss_G:{_loss_str}"
        logger.append(_log+'\n')
        pbar.set_description(_log)
        
    writer.add_images(f'train/img', torch.cat([outputs, targets], dim=-2)[::4], epoch * eval_step + i)
    avg_loss = loss_sum / eval_step
    _log = "==> [Eval] Epoch {} ({}), training loss={}".format(epoch, timestamp, avg_loss)
    
    if avg_loss < best_score:
        patience_cur = args.patience
        best_score = avg_loss        
        torch.save({'state_dict': model.state_dict()}, model_path)
        _log += '\n Found new best model!\n'
    else:
        patience_cur -= 1
        
    print(_log)
    with open(os.path.join(log_save_path, f'{task}_{timestamp}.log'), "a+") as log_file:
        log_file.writelines(logger)
    return avg_loss

if __name__ == '__main__':
    import time
    import os
    import torch
    from choose_character import character_choice
    from utils.common import parse_args_from_yaml, setup_seed, init_weights
    from models.DCGAN import Generator
    import torchvision.transforms as transforms
    import torch.nn as nn
    from dataset.ABAWData import ABAWDataset2
    from torch.utils.data import DataLoader
    from torch.optim import lr_scheduler
    from torch.utils.tensorboard import SummaryWriter
    task = 'rig2img'
    args = parse_args_from_yaml(f'configs_{task}.yaml')
    setup_seed(args.seed)
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    os.system("git add .")
    os.system("git commit -m" + timestamp)
    os.system("git push")
    
    configs_character = character_choice(args.character)
    mouth_crop = torch.tensor(configs_character['mouth_crop']).cuda().float()

    model_path = os.path.join(args.save_root,'ckpt', f"{task}_{timestamp}.pt")
    params = {'nz': configs_character['n_rig'], 'ngf': 64*2, 'nc': 3}
    model = Generator(params)
    model = model.cuda()
    
    if args.pretrained:
        ckpt_pretrained = os.path.join(args.save_root, 'ckpt', f"{task}_{args.pretrained}.pt")
        checkpoint = torch.load(ckpt_pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        print("load pretrained model {}".format(ckpt_pretrained))
    else:
        model.apply(init_weights)
        print("Model initialized")      
    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor()])
    
    criterion_l1 = nn.L1Loss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.0, 0.99))
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, 2, 1e-6)
 
    train_dataset = ABAWDataset2(root_path=configs_character['data_path'],character=args.character, only_render=True,
                                 data_split='train', transform=transform, return_rigs=True, n_rigs=configs_character['n_rig'])
    test_dataset = ABAWDataset2(root_path=configs_character['data_path'],character=args.character,only_render=True,
                                data_split='test', transform=transform, return_rigs=True, n_rigs=configs_character['n_rig'])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,num_workers=12)
    val_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True, num_workers=12)

    ck_save_path = f'{args.save_root}/ckpt'
    pred_save_path = f'{args.save_root}/test'
    log_save_path = f'{args.save_root}/logs'
    tensorboard_path = f'{args.save_root}/tensorboard/{timestamp}'
    
    os.makedirs(ck_save_path,exist_ok=True)
    os.makedirs(pred_save_path, exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)
    os.makedirs(log_save_path, exist_ok=True)

    writer = SummaryWriter(log_dir=tensorboard_path)
    
    patience_cur = args.patience
    best_score = float('inf')


    for epoch in range(500000000):
        avg_loss = Train(epoch, train_dataloader, model)
        avg_loss_eval = Eval(epoch, val_dataloader, model, best_score)
