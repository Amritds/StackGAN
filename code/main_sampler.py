from __future__ import print_function
import torch.backends.cudnn as cudnn
import torch
import torchvision.transforms as transforms

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil
import dateutil.tz

from StackGAN.miscc.datasets import TextDataset
from StackGAN.miscc.config import cfg, cfg_from_file
from StackGAN.miscc.utils import mkdir_p
from StackGAN.trainer import GANTrainer


cfg_from_file('cfg/coco_eval.yml')
cfg.GPU_ID ='0'
cfg.DATA_DIR = ''
    
print('Using config:')
pprint.pprint(cfg)
    
manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

if cfg.CUDA:
    torch.cuda.manual_seed_all(manualSeed)
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
output_dir = '../output/%s_%s_%s' % \
             (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

num_gpu = len(cfg.GPU_ID.split(','))

datapath= '%s' % (cfg.DATA_PATH)
algo = GANTrainer(output_dir)

def sample():
    return algo.sample(datapath, cfg.STAGE)
