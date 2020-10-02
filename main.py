import argparse
import os
import random
import time
import progressbar

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from vae_lp import VAE_LP
from ae_gan import AE_GAN
from data import DataLoader
import utils

args = utils.load_params(json_file='params.json')

data_loader = DataLoader(args)
vae = VAE_LP(args)
gan = AE_GAN(args)

for i in range(args['vae_epoch']):
    vae.train(data_loader.train_loader)
    if i % args['test_freq'] == 0:
        vae.test(data_loader.test_loader)
        vae.save_model('path')

for i in range(args['gan_epoch']):
    gan.train(vae, data_loader.train_loader)
    if i % args['test_freq'] == 0:
        gan.test(vae, data_loader.test_loader)
        gan.save_model('path')