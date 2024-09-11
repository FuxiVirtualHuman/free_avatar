from dataset import build_dataset
from model.mae_pipeline import Pipeline 
from torch import optim
from utils.metrics import triplet_prediction_accuracy
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.distance import PairwiseDistance
from tqdm import tqdm
import yaml
import argparse
import utils.misc as misc
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import math
import os

def read_yaml_to_dict(yaml_path):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value


def compute_loss(anc_fea, pos_fea, neg_fea, type):
    criterion1 = nn.TripletMarginLoss(margin=0.1)
    criterion2 = nn.TripletMarginLoss(margin=0.2)
    criterion3 = nn.TripletMarginLoss(margin=0.1)
    l2_dist = PairwiseDistance(2)
    loss = 0
    for i in range(len(type)):
        anc_ = anc_fea[i].unsqueeze(0)
        pos_ = pos_fea[i].unsqueeze(0)
        neg_ = neg_fea[i].unsqueeze(0)

        if type[i] == "ONE_CLASS_TRIPLET":
            loss += criterion1(anc_, pos_, neg_) + criterion1(pos_, anc_, neg_)
        else:
            loss += criterion2(anc_, pos_, neg_) + criterion2(pos_, anc_, neg_)

    loss = loss/anc_fea.shape[0]
    dists1 = l2_dist.forward(anc_fea, pos_fea).data.cpu().numpy()
    dists2 = l2_dist.forward(anc_fea, neg_fea).data.cpu().numpy()
    dists3 = l2_dist.forward(pos_fea, neg_fea).data.cpu().numpy()
    return loss,dists1,dists2,dists3



def main(config):
    trainset = build_dataset(config, "train")
    valset = build_dataset(config, "val")
    trainloader = DataLoader(trainset, batch_size=config["batch_size"],num_workers=config["num_workers"],shuffle=True)
    valloader = DataLoader(valset, batch_size=config["batch_size"],num_workers=config["num_workers"],shuffle=True)

    model = Pipeline(config).cuda()

    if config["resume"] != None:
        state_dict = torch.load(config["resume"])
        model.load_state_dict(state_dict)
    
    if config["use_dp"] == True:
        gpus = config["device"]
        model = torch.nn.DataParallel(model, device_ids=gpus)

    num_epochs = config["num_epochs"]

    if config["optim"] == "SGD":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"], momentum=config["momentum"],
                      weight_decay=config["weight_decay"])
    elif config["optim"] == "AdamW":
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"], betas=(0.9,0.999),
                      weight_decay=config["weight_decay"])

    metric_logger = misc.MetricLogger(delimiter="  ")
    test_metric_logger = misc.MetricLogger(delimiter="  ")
    log_writer = SummaryWriter(log_dir=config["log_dir"])

    os.makedirs(config["log_dir"],exist_ok=True)
    os.makedirs(config["checkpoint_dir"],exist_ok=True)
    

    for epoch in range(num_epochs):
        train_one_epoch(epoch, model, trainloader, metric_logger, log_writer, optimizer, config)
        evaluate(epoch, model, valloader, test_metric_logger, config)

    

def train_one_epoch(epoch, model, data_loader, metric_logger, log_writer, optimizer, config):
    model.train(True)
    print_freq = config["print_freq"]
    accum_iter = config["accum_iter"]
    header = 'Training Epoch: [{}]'.format(epoch)
    t = enumerate(metric_logger.log_every(data_loader, print_freq, header))
    for step, samples in t:
        anc_img, pos_img, neg_img, anc_list, type = samples["anc"],samples["pos"],samples["neg"],samples["name"],samples["type"]
        anc_img, pos_img, neg_img = anc_img.cuda(), pos_img.cuda(), neg_img.cuda()
        model.zero_grad()
        vec = torch.cat((anc_img, pos_img, neg_img), dim=0)
        emb = model.forward(vec)
        ll = int(emb.shape[0] / 3)
        anc_fea, pos_fea, neg_fea = torch.split(emb, ll, dim=0)

        loss,dists1,dists2,dists3 = compute_loss(anc_fea, pos_fea, neg_fea, type)
        
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.state_dict()['param_groups'][0]['lr'])

        loss_value_reduce = misc.all_reduce_mean(loss_value)


        if (step + 1) % accum_iter == 0:
            iter = epoch * len(data_loader) + step + 1
            log_writer.add_scalar("loss", loss_value, iter)

    print("Averaged stats:", metric_logger)


def evaluate(epoch, model, data_loader, test_metric_logger, config):
    model.eval()
    print_freq = config["print_freq"]
    header = 'Validation Epoch: [{}]'.format(epoch)
    acc_logger = misc.Triplet_Logger(os.path.join(config["log_dir"], "test_log.json"))

    t = enumerate(test_metric_logger.log_every(data_loader, print_freq, header))

    for step, samples in t:
        anc_img, pos_img, neg_img, anc_list, type = samples["anc"],samples["pos"],samples["neg"],samples["name"],samples["type"]
        anc_img, pos_img, neg_img = anc_img.cuda(), pos_img.cuda(), neg_img.cuda()

        vec = torch.cat((anc_img, pos_img, neg_img), dim=0)
        with torch.no_grad():
            emb = model.forward(vec)
            ll = int(emb.shape[0] / 3)
            anc_fea, pos_fea, neg_fea = torch.split(emb, ll, dim=0)

        loss,dists1,dists2,dists3 = compute_loss(anc_fea, pos_fea, neg_fea, type)
        loss_value = loss.item()
        
        test_metric_logger.update(loss=loss_value)
        acc_logger.update(dists1,dists2,dists3,loss,type)
    
    avg_loss, res = acc_logger.summary()
    print(res)
    test_metric_logger.meters['loss_avg'].update(avg_loss, n=1)
    test_metric_logger.meters['overall_accuracy'].update(res[0], n=1)

    if len(res)>1:
        test_metric_logger.meters['CLASS1_1_accuracy'].update(res[1], n=1)
        test_metric_logger.meters['CLASS1_2_accuracy'].update(res[2], n=1)
        test_metric_logger.meters['CLASS1_3_accuracy'].update(res[3], n=1)
    
    print('* Overall Accuracy: {overall_accuracy.avg:.3f}  loss {loss_avg.global_avg:.3f}'
        .format(overall_accuracy = test_metric_logger.overall_accuracy, loss_avg = test_metric_logger.meters["loss_avg"]))
    
    if epoch % config["save_epoch"] == 0:
        save_path = os.path.join(config["checkpoint_dir"], "epoch_" + str(epoch) + "_acc_" + str(res[0]) + ".pth")
        torch.save(model.state_dict(),save_path)
    
    return avg_loss, res



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mae_train_expemb.yaml")
    args = parser.parse_args()
    yml_path = args.config
    config = read_yaml_to_dict(yml_path)

    main(config)