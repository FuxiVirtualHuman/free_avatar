import cv2
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
import torchvision.transforms as transforms
from utils.common import *
from models.DCGAN import Generator
from models.gan_loss import GANLoss

def Train(epoch, loader, model, model_D):
    lr = optimizer.param_groups[0]['lr']
    print(f"*** Epoch {epoch}, lr:{lr:.5f}, timestamp:{timestamp}")
    loss_sum = 0.0
    loss_sum_D = 0.
    model_D = model_D.train()
    train_step = min(args.train_step_per_epoch, len(loader))
    b = '{l_bar}{bar:20}{r_bar}{bar:10b}'
    pbar = tqdm(enumerate(loader), bar_format=b, total=train_step)
    logger = []
    time0 = time.time()
    for i, data in pbar:
        if i>train_step:
            break
        optimizer.zero_grad()
        model.train()
        model_D.eval()
        loss = dict()

        sources = data['img'].cuda().float()
        targets = data['target'].cuda().float()
        target_rigs = data['rigs'].cuda().float()
        is_render = data['is_render'].cuda().float()
        ldmk = data['ldmk'].cuda().float()
        role_id = data['role_id'].cuda().long()
        do_pixel = data['do_pixel'].cuda().int()
        bs_input = data['bs'].cuda().float() 
        has_rig = data['has_rig'].cuda().float() 
        
        
        real_idx = ((is_render == 0).nonzero(as_tuple=True)[0])
        render_idx = ((is_render == 1).nonzero(as_tuple=True)[0])
        has_rig = ((has_rig == 1).nonzero(as_tuple=True)[0])
        do_pixel_idx = ((do_pixel == 1).nonzero(as_tuple=True)[0])
        
        role_idxes = [((role_id == CHARACTER_NAMES.index(name_e)).nonzero(as_tuple=True)[0]) for name_e in characters]
        
        # source image to emb
        with torch.no_grad():
            emb_hidden_in, emb_in = model_emb(resize(sources))
            if args.weight_symm:
                emb_hidden_symm_in, emb_symm_in = model_symm(resize_symm(sources))
                emb_hidden_in = torch.cat((emb_hidden_in, emb_hidden_symm_in), dim=1)
            
            
        output_rig = model(emb_hidden_in, role_id)
        
        with torch.no_grad():
            output_imgs_c = []
            for c_i, cname in enumerate(characters):
                output_imgs_c.append(configs_characters[cname]["model_rig2img"](output_rig[role_idxes[c_i]][:,:configs_characters[cname]['n_rig']].reshape(-1, configs_characters[cname]['n_rig'], 1, 1)))
            
            C, H, W = output_imgs_c[0].shape[1:]
            B = sources.shape[0]
            output_img = torch.empty((B, C, H, W), dtype=torch.float32).cuda() 
            for c_i, cname in enumerate(characters):
                output_img[role_idxes[c_i]] = output_imgs_c[c_i]
            
        if args.weight_rig:
            if len(do_pixel_idx)>0:
                loss['rig'] = criterion_l2(output_rig[has_rig], target_rigs[has_rig])
        
        if args.weight_emb:
            emb_hidden_out, emb_out = model_emb(resize(output_img))
            loss['emb']  = criterion_l2(emb_out, emb_in)

        if args.weight_img:
            if len(do_pixel_idx) > 0:
                loss['image'] = criterion_l1(output_img[do_pixel_idx], targets[do_pixel_idx]) * args.weight_img

        if args.weight_D:
            output_D_G = model_D(output_img)
            loss['G_D'] = criterion_gan(output_D_G, True, False)[0] * args.weight_D

        if args.weight_symm:
            with torch.no_grad():
                emb_hidden_symm_out, emb_symm_out = model_symm(resize_symm(output_img))
            loss['symm'] =  criterion_l2(emb_symm_out, emb_symm_in) * args.weight_symm
        if not loss:
            continue

        
        loss_value = sum([v for k, v in loss.items()])
        
        loss_sum += loss_value.item()
        loss_value.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        
        # discriminator
        optimizer_D.zero_grad()
        model_D.train()
        model.eval()
        outputs_fake = model_D(output_img.detach())
        outputs_real = model_D(targets[do_pixel_idx])
        loss_fake = criterion_gan(outputs_fake, False, True)[0]
        loss_real = criterion_gan(outputs_real, True, True)[0]
        loss_D_train = (loss_fake + loss_real) / 2.
        loss_D_train.backward()
        optimizer_D.step()
        scheduler_D.step()

        writer.add_scalars(f'train/loss_G', loss, epoch * train_step + i)
        writer.add_scalar(f'train/loss_G_total', loss_value.item(), epoch * train_step + i)
        writer.add_scalar(f'train/loss_D', loss_D_train.item(), epoch * train_step + i)
        
        _loss_str = str({k: "{:.4f}".format(v/(i+1)) for k, v in loss.items()})
        _log = f"Epoch {epoch}({timestamp}) (lr:{optimizer.param_groups[0]['lr']:05f}): [{i}/{len(train_dataloader)}] loss_G:{_loss_str}, loss_D: {loss_D_train.item():.04f}"
        logger.append(_log+'\n')
        pbar.set_description(_log)
        
    writer.add_images(f'train/img', torch.cat([sources, targets, output_img], dim=-2)[::4], epoch * train_step + i)
    avg_loss = loss_sum / train_step
    _log = "==> [Train] Epoch {} ({}), training loss={}".format(epoch, timestamp, avg_loss)
    print(_log)
    with open(os.path.join(log_save_path, f'emb2render_{timestamp}.log'), "a+") as log_file:
        log_file.writelines(logger)
    if epoch % args.save_step == 0:
        torch.save({'state_dict': model.state_dict()}, model_path.replace('.pt', f'_{epoch}.pt'))
        torch.save({'state_dict': model_D.state_dict()}, model_path.replace('.pt', f'_{epoch}_D.pt'))
    return avg_loss

def Eval(epoch, loader, model, model_D, best_score):
    loss_sum = 0.0
    model = model.eval()
    model_D = model_D.eval()

    eval_step = min(args.eval_step_per_epoch, len(loader))
    b = '{l_bar}{bar:20}{r_bar}{bar:10b}'
    pbar = tqdm(enumerate(loader), bar_format=b, total=eval_step)
    logger = []
    time0 = time.time()
    for i, data in pbar:
        if i>eval_step:
            break
        loss = dict()

        sources = data['img'].cuda().float()
        targets = data['target'].cuda().float()
        target_rigs = data['rigs'].cuda().float()
        is_render = data['is_render'].cuda().float()
        ldmk = data['ldmk'].cuda().float()
        role_id = data['role_id'].cuda().long()
        do_pixel = data['do_pixel'].cuda().int()
        bs_input = data['bs'].cuda().float() 
        
        
        real_idx = ((is_render == 0).nonzero(as_tuple=True)[0])
        render_idx = ((is_render == 1).nonzero(as_tuple=True)[0])
        do_pixel_idx = ((do_pixel == 1).nonzero(as_tuple=True)[0])
        role_idx0 = ((role_id == 0).nonzero(as_tuple=True)[0])
        role_idx1 = ((role_id == 1).nonzero(as_tuple=True)[0])
        role_idxes = [((role_id == CHARACTER_NAMES.index(name_e)).nonzero(as_tuple=True)[0]) for name_e in characters]
        # source image to emb
        with torch.no_grad():
            
            emb_hidden_in, emb_in = model_emb(resize(sources))
            if args.weight_symm:
                emb_hidden_symm_in, emb_symm_in = model_symm(resize_symm(sources))
                emb_hidden_in = torch.cat((emb_hidden_in, emb_hidden_symm_in), dim=1)
            output_rig = model(emb_hidden_in, role_id)
            output_imgs_c = []
            for c_i, cname in enumerate(characters):
                output_imgs_c.append(configs_characters[cname]["model_rig2img"](output_rig[role_idxes[c_i]][:,:configs_characters[cname]['n_rig']].reshape(-1, configs_characters[cname]['n_rig'], 1, 1)))
            
            C, H, W = output_imgs_c[0].shape[1:]
            B = sources.shape[0]
            output_img = torch.empty((B, C, H, W), dtype=torch.float32).cuda()  # 假设输出的类型为float32
            for c_i, cname in enumerate(characters):
                output_img[role_idxes[c_i]] = output_imgs_c[c_i]
            
        if args.weight_rig:
            if len(do_pixel_idx)>0:
                loss['rig'] = criterion_l2(output_rig[do_pixel_idx], target_rigs[do_pixel_idx])
        
        if args.weight_emb:
            emb_hidden_out, emb_out = model_emb(resize(output_img))
            loss['emb']  = criterion_l2(emb_out, emb_in)

        if args.weight_img:
            if len(do_pixel_idx) > 0:
                loss['image'] = criterion_l1(output_img[do_pixel_idx], targets[do_pixel_idx]) * args.weight_img
        
        if args.weight_D:
            output_D_G = model_D(output_img)
            loss['G_D'] = criterion_gan(output_D_G, True, False)[0] * args.weight_D
            
        if args.weight_symm:
            with torch.no_grad():
                emb_hidden_symm_out, emb_symm_out = model_symm(resize_symm(output_img))
            loss['symm'] =  criterion_l2(emb_symm_out, emb_symm_in) * args.weight_symm
            
        if not loss:
            continue
        
        loss_value = sum([v for k, v in loss.items()])
        
        loss_sum += loss_value.item()

        writer.add_scalars(f'eval/loss_G', loss, epoch * eval_step + i)
        writer.add_scalar(f'eval/loss_G_total', loss_value.item(), epoch * eval_step + i)
        
        _loss_str = str({k: "{:.4f}".format(v/(i+1)) for k, v in loss.items()})
        _log = f"Epoch {epoch}({timestamp}) (lr:{optimizer.param_groups[0]['lr']:05f}): [{i}/{len(train_dataloader)}] loss_G:{_loss_str}"
        logger.append(_log+'\n')
        pbar.set_description(_log)
        
    writer.add_images(f'eval/img', torch.cat([sources, targets, output_img], dim=-2)[::4], epoch * eval_step + i)
    avg_loss = loss_sum / eval_step
    _log = "==> [Eval] Epoch {} ({}), evaluation loss={}".format(epoch, timestamp, avg_loss)

    
    if avg_loss < best_score:
        patience_cur = args.patience
        best_score = avg_loss        
        torch.save({'state_dict': model.state_dict()}, model_path)
        torch.save({'state_dict': model_D.state_dict()}, model_path.replace('.pt', f'_D.pt'))
        _log += '\n Found new best model!\n'
    else:
        patience_cur -= 1
    print(_log)
    logger.append(_log)
    with open(os.path.join(log_save_path, f'emb2render_{timestamp}.log'), "a+") as log_file:
        log_file.writelines(logger)  
    return avg_loss

def Test(signature, model, model_emb, model_rig2img, resize):
    model = model.eval()
    save_root ='/project/qiuf/expr-capture/test'
    root = '/data/Workspace/Rig2Face/data'
    folders = ['ziva']
    
    for fold in folders:
        save_fold = os.path.join(save_root, f'{signature}_{fold}')
        os.makedirs(save_fold, exist_ok=True)
        imgnames = os.listdir(os.path.join(root, fold))
        imgnames.sort()
        imgnames = imgnames[:]

        rigs = {}
        for charname in characters:
            rigs[charname] = []
        for i, img in tqdm(enumerate(imgnames), total=len(imgnames)):
            img = cv2.resize(cv2.imread(os.path.join(root, fold, img)), (256,256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.FloatTensor(img).permute(2,0,1).unsqueeze(0).cuda()/255.
            
            with torch.no_grad():
                emb_hidden, emb = model_emb(resize(img_tensor))
                if args.weight_symm:
                    emb_hidden_symm_in, emb_symm_in = model_symm(resize_symm(img_tensor))
                    emb_hidden = torch.cat((emb_hidden, emb_hidden_symm_in), dim=1)
                
                img_outs=  []
                emb_dists = []
                for c_i, cname in enumerate(characters):
                    rig = model(emb_hidden, torch.LongTensor([CHARACTER_NAMES.index(cname),]).cuda())
                    rigs[cname].append(rig.cpu().numpy())
                    img_outs.append(configs_characters[cname]["model_rig2img"](rig[:, :configs_characters[cname]['n_rig']].reshape(1,-1,1,1)))
                    emb_hidden_out0, emb_out0 = model_emb(resize(img_outs[-1]))
                    emb_dists.append(torch.dist(emb, emb_out0))
                

                img_vis = torch.cat((img_tensor, *img_outs), dim=-1).squeeze()*255. 
                img_vis = img_vis.cpu().numpy().transpose(1,2,0).astype(np.uint8)
                img_vis = np.ascontiguousarray(img_vis[...,::-1], dtype=np.uint8)
                for c_i, cname in enumerate(characters):
                    cv2.putText(img_vis, str(np.round(emb_dists[c_i].item(), 6)), (256*(c_i+1), 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(save_fold, f'{i:05d}.jpg'), img_vis)
                
        for c_i, cname in enumerate(characters):
            np.savetxt(os.path.join(save_root, f'ziva_{cname}.txt'), np.array(rigs[cname]).squeeze())

        imgs2video(save_fold)


if __name__ == '__main__':
    import time
    from choose_character import character_choice
    from models.load_emb_model import load_emb_model
    from models.CascadeNet import get_model
    from models.discriminator import MultiscaleDiscriminator, get_parser
    from dataset.ABAWData import ABAWDataset2_multichar
    from torch.utils.tensorboard import SummaryWriter
    args = parse_args_from_yaml('configs_emb2rig_multi.yaml')
    setup_seed(args.seed)
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    os.system("git add .")
    os.system("git commit -m" + timestamp)
    os.system("git push")
    
    # emb_model
    characters = args.character.replace(' ','').split(',')
    CHARACTER_NAMES = args.CHARACTER_NAME
    configs_characters = {e:character_choice(e) for e in characters}
    n_rig = max([e['n_rig'] for e in configs_characters.values()])
    for character in configs_characters:
        configs_characters[character]['mouth_crop'] = torch.tensor(configs_characters[character]['mouth_crop']).cuda().float()
        params = {'nz': configs_characters[character]['n_rig'], 'ngf': 64*2, 'nc': 3}
        model_rig2img = Generator(params)
        model_rig2img = model_rig2img.eval().cuda()
        ckpt_generator = torch.load(configs_characters[character]['ckpt_rig2img'])
        model_rig2img.load_state_dict(ckpt_generator['state_dict'])
        configs_characters[character]['model_rig2img'] = model_rig2img
        print('load generator model from {}'.format(configs_characters[character]['ckpt_rig2img']))
        
    model_emb, emb_dim, resize = load_emb_model(args.emb_backbone)
    model_emb = model_emb.eval().cuda()
    model_emb_params = count_parameters(model_emb)
    print('emb model:', model_emb_params)
    
    # dissymm model 
    if args.weight_symm:
        model_symm, emb_dim2, resize_symm = load_emb_model('dissymm_repvit')
        model_symm.cuda().eval()
        emb_dim += emb_dim2

    # img2rig model
    pass

    # emb2rig model
    model_path = os.path.join(args.save_root, 'ckpt', "emb2rig_multi_{}.pt".format(timestamp))
    model = get_model(1, refine_3d=False,
                                 norm_twoD=False,
                                 num_blocks=2, #5,
                                 input_size=emb_dim,
                                 output_size=n_rig,
                                 linear_size=512, #1024,
                                 dropout=0.1,
                                 leaky=False,
                                 use_multichar=args.use_multichar,
                                 id_embedding_dim=args.id_embedding_dim
                                 )
    model = model.cuda()
    model_params = count_parameters(model)
    print('emb2rig model:', model_params)
    
    # D_model
    opt = get_parser()
    model_D = MultiscaleDiscriminator(opt).cuda()

    if args.pretrained:
        ckpt_pretrained = os.path.join(args.save_root, 'ckpt', "emb2rig_multi_{}.pt".format(args.pretrained))
        ckpt_pretrained_D = os.path.join(args.save_root, 'ckpt', "emb2rig_multi_{}_D.pt".format(args.pretrained))
        checkpoint = torch.load(ckpt_pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        checkpoint_D = torch.load(ckpt_pretrained_D)
        model_D.load_state_dict(checkpoint_D['state_dict'])

        print("load pretrained model {}".format(ckpt_pretrained))
    else:
        model.apply(init_weights)
        model_D.apply(init_weights)
        print("Model initialized")         
    
    # transforms
    transform1 = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.Resize([256,256]),
        transforms.ToTensor(),
    ])
    
    transform2 = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])
    
    # optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.0, 0.99))
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, 2, 1e-6)
    optimizer_D = torch.optim.Adam(filter(lambda p: p.requires_grad, model_D.parameters()), lr=args.lr_D, betas=(0.0, 0.99), weight_decay=1e-6)
    scheduler_D = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_D, 500, 2, 1e-7)
    
    # loss function 
    criterion_l1 = nn.L1Loss()
    criterion_l2 = nn.MSELoss()
    criterion_BCE = nn.BCELoss()
    criterion_gan = GANLoss('hinge')
    
    # Test
    if args.mode == 'test':
        Test(args.pretrained, model, model_emb, model_rig2img, resize)
        exit()
    
    # datasets
    train_dataset = ABAWDataset2_multichar(configs_characters, data_split='train',CHARACTER_NAME=CHARACTER_NAMES, transform=transform1, return_rigs=True)
    test_dataset = ABAWDataset2_multichar(configs_characters, data_split='test',CHARACTER_NAME=CHARACTER_NAMES, transform=transform2, return_rigs=True)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,num_workers=8)
    val_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True, num_workers=8)
    
    # save files
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
        avg_loss = Train(epoch, train_dataloader, model, model_D)
        avg_loss_eval = Eval(epoch, val_dataloader, model, model_D, best_score)
        if epoch % args.save_step == 0:
            Test(timestamp+'_'+str(epoch), model, model_emb, model_rig2img, resize)
