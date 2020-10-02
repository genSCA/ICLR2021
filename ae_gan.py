import argparse
import os
import random
import time
import progressbar

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models import TraceEncoder, ImageEncoder, ImageDecoder
from models import Generator as G
from models import Discriminator as D
import utils

class AE_GAN(object):
    def __init__(self, args):
        self.args = args
        self.epoch = 0
        self.bce = nn.BCELoss().cuda()
        self.real_label = torch.FloatTensor(1).cuda().fill_(1)
        self.fake_label = torch.FloatTensor(1).cuda().fill_(0)
        self.init_model_optimizer()

    def loss(self, output, label):
        return self.bce(output, label.expand_as(output))

    def init_model_optimizer(self):
        print('Initializing Model & Optimizer...')
        self.G = G(nc=self.args['nc'], dim=self.args.['gan_dim'])
        self.G = torch.nn.DataParallel(self.G).cuda()
        self.optimizerG = torch.optim.Adam(
            self.G.module.parameters(),
            lr=self.args.['gan_lr'], betas=(self.args.['beta1'], 0.999)
            )
        self.D = D(nc=self.args['nc'], dim=self.args.['gan_dim'])
        self.D = torch.nn.DataParallel(self.D).cuda()
        self.optimizerD = torch.optim.Adam(
            self.D.module.parameters(),
            lr=self.args.['gan_lr'], betas=(self.args.['beta1'], 0.999)
            )

    def load_model(self, path):
        print('Loading Model from %s ...' % (path))
        ckpt = torch.load(path)
        self.G.module.load_state_dict(checkpoint['Generator'])
        self.D.module.load_state_dict(checkpoint['Discriminator'])

    def save_model(self, path):
        print('Saving Model on %s ...' % (path))
        state = {
            'Generator': self.G.module.state_dict(),
            'Discriminator': self.D.module.state_dict()
        }
        torch.save(state, path)

    def train(self, VAE_LP, data_loader):
        print('Training...')
        with torch.autograd.set_detect_anomaly(True):
            self.epoch += 1
            self.G.train()
            self.D.train()
            record_G = utils.Record()
            record_D = utils.Record()
            current_time = time.time()
            progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
            for i, (trace, image) in enumerate(data_loader):
                progress.update(i + 1)
                trace = trace.cuda()
                image = image.cuda()
                
                self.D.zero_grad()
                
                real_output = self.D(image)
                
                err_D_real = self.loss(real_output, self.real_label)
                err_D_real.backward(retain_graph=True)
                D_x = real_output.data.mean()

                fake_input, *_ = VAE_LP.inference(trace)
                fake_refine = self.G(fake_input)
                fake_output = self.D(fake_refine.detach())
                
                err_D_fake = self.loss(fake_output, self.fake_label)
                err_D_fake.backward(retain_graph=True)
                D_G_z = fake_output.data.mean()

                err_D = err_D_fake + err_D_real
                optimizerD.step()

                self.G.zero_grad()
                fake_output = self.D(fake_refine)
                err_G = self.loss(fake_output, self.real_label)
                err_G.backward()
                optimizerG.step()

                record_D.add(err_D.item())
                record_G.add(err_G.item())
            progress.finish()
            utils.clear_progressbar()
            print('----------------------------------------')
            print('Epoch: %d' % epoch)
            print('Costs time: %.2f s' % (time.time() - current_time))
            print('Error of G is: %f' % (record_G.mean()))
            print('Error of D is: %f' % (record_D.mean()))
            print('D(x) is: %f, D(G(z)) is: %f' % (D_x, D_G_z))

            utils.save_image(image.data, (self.args['gan_path']+'/image/train/target_%3d.jpg') % epoch)
            utils.save_image(fake_input.data, (self.args['gan_path']+'/image/train/tr2im_%3d.jpg') % epoch)
            utils.save_image(fake_refine.data, (self.args['gan_path']+'/image/train/final_%3d.jpg') % epoch)

    def test(self, VAE_LP, data_loader):
        print('Testing...')
        with torch.no_grad():
            self.G.eval()
            self.D.eval()
            record_G = utils.Record()
            record_D = utils.Record()
            current_time = time.time()
            progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
            for i, (trace, image) in enumerate(data_loader):
                progress.update(i + 1)
                trace = trace.cuda()
                image = image.cuda()
                
                real_output = self.D(image)
                err_D_real = self.loss(real_output, self.real_label)
                D_x = real_output.data.mean()

                fake_input, *_ = VAE_LP.inference(trace)
                fake_refine = self.G(fake_input)
                fake_output = self.D(fake_refine.detach())
                err_D_fake = self.loss(fake_output, self.fake_label)
                D_G_z = fake_output.data.mean()

                err_D = err_D_fake + err_D_real

                fake_output = self.D(fake_refine)
                err_G = self.loss(fake_output, self.real_label)

                record_D.add(err_D.item())
                record_G.add(err_G.item())
            progress.finish()
            utils.clear_progressbar()
            print('----------------------------------------')
            print('Test at Epoch %d' % epoch)
            print('Costs time: %.2f s' % (time.time() - current_time))
            print('Error of G is: %f' % (record_G.mean()))
            print('Error of D is: %f' % (record_D.mean()))
            print('D(x) is: %f, D(G(z)) is: %f' % (D_x, D_G_z))

            utils.save_image(image.data, (self.args['gan_path']+'/image/test/target_%3d.jpg') % epoch)
            utils.save_image(fake_input.data, (self.args['gan_path']+'/image/test/tr2im_%3d.jpg') % epoch)
            utils.save_image(fake_refine.data, (self.args['gan_path']+'/image/test/final_%3d.jpg') % epoch)

    def inference(self, VAE_LP, x):
        with torch.no_grad():
            self.G.eval()
            recov_image = VAE_LP.inference(x)
            final_image = self.G(recov_image)
        return final_image