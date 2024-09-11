import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist
from torch._six import inf
from utils.metrics import triplet_prediction_accuracy
import json
import os
import pickle

class SmoothedValue(object):
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter


    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))



class Triplet_Logger:
    def __init__(self, save_log_file=None):
        self.distance1 = []
        self.distance2 = []
        self.distance3 = []
        self.losses = []
        self.types = []
        self.save_log_file = save_log_file
 

    def update(self, dis1, dis2, dis3, loss, type):
        self.distance1.extend(dis1)
        self.distance2.extend(dis2)
        self.distance3.extend(dis3)
        self.losses.append(loss)
        self.types.extend(type)

    def dict_to_str(self,dict):
        logs = ""
        for k,v in dict.items():
            logs += k + ":  " +str(v) + " "
        return logs

    def write_log(self, avg_loss, res):
        if len(res)==1:
            write_json_dict = {'time':time.asctime(), 'Avg losses': avg_loss.item(), 'Overall prediction Accuracy': res[0]}
        else:
            write_json_dict = {'time':time.asctime(), 'Avg losses': avg_loss.item(), 'Overall prediction Accuracy': res[0], \
             'Class 1 Acc': res[1], 'Class 2 Acc': res[2], 'Class 3 Acc': res[3]}
        logs = self.dict_to_str(write_json_dict)
        with open(self.save_log_file, mode="a") as f:
            f.write(logs + "\n")

       
    def summary(self):
        res = []
        avg_loss = sum(self.losses)/len(self.losses)
        
        acc, acc1, acc2, acc3 = triplet_prediction_accuracy(self.distance1,self.distance2,self.distance3,self.types,"triplet")
        res = [acc, acc1, acc2, acc3]
        
        if self.save_log_file!=None:
            self.write_log(avg_loss,res)
        return avg_loss, res




class Emb_Logger:
    def __init__(self,save_path=None):
        self.emb_dict = {}
        self.save_path = save_path
 
    def update(self, names , embs):
        for i in range(len(names)):
            self.emb_dict[names[i]] = embs[i]

    def summary(self):
        
        with open(self.save_path, mode="wb") as f:
                pickle.dump(self.emb_dict,f)
        
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x