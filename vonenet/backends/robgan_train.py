import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import torch
import torch.nn as nn
from torch.optim import Adam
import time

from torch.autograd import Variable
import torchvision.transforms as tfs
from torchvision import datasets
from torch.utils.data import DataLoader

import numpy as np

from miscs.pgd import attack_Linf_PGD, attack_FGSM
from miscs.loss import *

def noise(x):
    return x + torch.FloatTensor(x.size()).uniform_(0, 1.0 / 128)
def make_dataset(img_size, dataset='mnist', root='./assets/data/mnist', batch_size=64, num_workers=2):
    # Small noise is added, following SN-GAN
    if dataset == "cifar10":
        trans = tfs.Compose([
            # tfs.RandomCrop(opt.img_width, padding=4),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
            tfs.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
            tfs.Lambda(noise)])
        data = datasets.CIFAR10(root=root, train=True, download=False, transform=trans)
        loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    elif dataset == 'mnist':
        dataset = datasets.MNIST(
            root=root,
            train=True,
            download=True,
            transform=tfs.Compose([
                tfs.Resize(img_size), #FIXME
                tfs.ToTensor(),
                tfs.Normalize([0.5], [0.5]),
                tfs.Lambda(noise),
            ]))

        loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)
    # elif opt.dataset == "dog_and_cat_64":
    #     trans = tfs.Compose([
    #         tfs.RandomResizedCrop(opt.img_width, scale=(0.8, 0.9), ratio=(1.0, 1.0)),
    #         tfs.RandomHorizontalFlip(),
    #         tfs.ToTensor(),
    #         tfs.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
    #         tfs.Lambda(noise)])
    #     data = ImageFolder(opt.root, transform=trans)
    #     loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    # elif opt.dataset == "dog_and_cat_128":
    #     trans = tfs.Compose([
    #         tfs.RandomResizedCrop(opt.img_width, scale=(0.8, 0.9), ratio=(1.0, 1.0)),
    #         tfs.RandomHorizontalFlip(),
    #         tfs.ToTensor(),
    #         tfs.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
    #         tfs.Lambda(noise)])
    #     data = ImageFolder(opt.root, transform=trans)
    #     loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    # elif opt.dataset == "imagenet":
    #     trans = tfs.Compose([
    #         tfs.RandomResizedCrop(opt.img_width, scale=(0.8, 0.9), ratio=(1.0, 1.0)),
    #         tfs.RandomHorizontalFlip(),
    #         tfs.ToTensor(),
    #         tfs.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
    #         tfs.Lambda(noise)])
    #     data = ImageFolder(opt.root, transform=trans)
    #     loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return loader

def get_loss():
    return loss_nll, loss_nll

class RobGANTrainer(object):

    def __init__(self, discriminator: nn.Module, generator: nn.Module, device: torch.device,
                 num_classes: int, lr: float, b1: float, b2: float, img_size: int,
                 batch_size: int, num_workers: int, latent_dim: int):
        self.name = 'robgan_train'
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.Ld, self.Lg = get_loss()
        self.lr = lr
        self.betas = (b1, b2)
        self.g_optimizer = Adam(self.generator.parameters(),
                                lr=self.lr, betas=self.betas)
        self.d_optimizer = Adam(self.discriminator.parameters(),  # Check parameters
                                lr=self.lr, betas=self.betas)

        self.img_size = img_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.latent_dim = latent_dim
        self.iter_d = 5
        self.adv_steps = 5
        self.epsilon = 0.0625

        self.data_loader = make_dataset(img_size = self.img_size, batch_size=self.batch_size, num_workers=self.num_workers)



    def get_sample(self, n_row=8):
        """Saves a grid of generated digits ranging from 0 to n_classes"""

        # Sample noise
        z = Variable(torch.FloatTensor(np.random.normal(
            0, 1, (n_row ** 2, self.latent_dim))).cuda())
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([num for _ in range(n_row) for num in range(n_row)])
        labels = Variable(torch.LongTensor(labels).cuda())
        gen_imgs = self.generator(z, labels)
        return gen_imgs

    def __call__(self, with_gen, inp, target):
        start = time.time()

        # gaussian noise
        z = torch.FloatTensor(self.batch_size, self.latent_dim).cuda()
        fixed_z = Variable(torch.FloatTensor(8 * 10, self.latent_dim).normal_(0, 1).cuda())
        # random label
        y_fake = torch.LongTensor(self.batch_size).cuda()
        np_y = np.arange(10)
        np_y = np.repeat(np_y, 8)
        fixed_y_fake = Variable(torch.from_numpy(np_y).cuda())
        # fixed label
        zeros = Variable(torch.FloatTensor(self.batch_size).fill_(0).cuda())
        ones = Variable(torch.FloatTensor(self.batch_size).fill_(1).cuda())
        # start training
        if with_gen:
            # update generator for every iter_d iterations
            self.generator.zero_grad()
            # sample noise
            z.normal_(0, 1)
            vz = Variable(z)
            y_fake.random_(0, to=self.num_classes)
            v_y_fake = Variable(y_fake)
            v_x_fake = self.generator(vz, y=v_y_fake)
            v_x_fake_adv = v_x_fake
            d_fake_bin, d_fake_multi = self.discriminator(v_x_fake_adv)
            with torch.no_grad():
                ones.resize_as_(d_fake_bin.data)
            loss_g = self.Lg(d_fake_bin, ones, d_fake_multi, v_y_fake, lam=0.5)
            loss_g.backward()
            self.g_optimizer.step()
            # print(f'[{epoch}/{opt.max_epoch-1}][{count+1}/{len(train_loader)}][G_ITER] loss_g: {loss_g.item()}')
        # update discriminator
        self.discriminator.zero_grad()
        # feed real data
        x_real, y_real = inp, target
        x_real, y_real = x_real.cuda(), y_real.cuda()
        v_x_real, v_y_real = Variable(x_real), Variable(y_real)
        # find adversarial example
        with torch.no_grad():
            ones.resize_(y_real.size())
        v_x_real_adv = attack_Linf_PGD(v_x_real, ones, v_y_real, self.discriminator, self.Ld, self.adv_steps, self.epsilon)
        d_real_bin, d_real_multi = self.discriminator(v_x_real_adv)
        # accuracy for real images
        positive = torch.sum(d_real_bin.data > 0).item()
        _, idx = torch.max(d_real_multi.data, dim=1)
        correct_real = torch.sum(idx.eq(y_real)).item()
        total_real = y_real.numel()
        # loss for real images
        loss_d_real = self.Ld(d_real_bin, ones, d_real_multi, v_y_real, lam=0.5)
        # feed fake data
        z.normal_(0, 1)
        y_fake.random_(0, to=self.num_classes)
        vz, v_y_fake = Variable(z), Variable(y_fake)
        with torch.no_grad():
            v_x_fake = self.generator(vz, y=v_y_fake)
        d_fake_bin, d_fake_multi = self.discriminator(v_x_fake.detach())
        # accuracy for fake images
        negative = torch.sum(d_fake_bin.data > 0).item()
        _, idx = torch.max(d_fake_multi.data, dim=1)
        correct_fake = torch.sum(idx.eq(y_fake)).item()
        total_fake = y_fake.numel()
        # loss for fake images
        loss_d_fake = self.Ld(d_fake_bin, zeros, d_fake_multi, v_y_fake, lam=1) # or lam=0.5
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        self.d_optimizer.step()
            # print(f'[{epoch}/{opt.max_epoch-1}][{count+1}/{len(train_loader)}][D_ITER] loss_d: {loss_d.item()} acc_r: {positive/total_real}, acc_r@1: {correct_real/total_real}, acc_f: {negative/total_fake}, acc_f@1: {correct_fake/total_fake}')
        # generate samples
        # with torch.no_grad():
        #     fixed_x_fake = self.generator(fixed_z, y=fixed_y_fake)
        #     fixed_x_fake.data.mul_(0.5).add_(0.5)
        # x_real.mul_(0.5).add_(0.5)
        # save_image(fixed_x_fake.data, f'./{opt.out_f}/sample_epoch_{epoch}.png', nrow=8)
        # save_image(x_real, f'./{opt.out_f}/real.png')

        record = {}
        record['d_loss_fake'] = loss_d_fake.item()
        record['d_loss_real'] = loss_d_real.item()
        if with_gen:
            record['g_loss'] = loss_g.item()
        record['acc_r'] = positive/total_real
        record['acc_r@1'] = correct_real/total_real
        record['acc_f'] = negative/total_fake
        record['acc_f@1'] = correct_fake/total_fake
        record['duration'] = time.time() - start

        return record

    def update_optimizers(self):
        self.lr /= 2
        self.g_optimizer = Adam(self.generator.parameters(),
                                lr=self.lr, betas=self.betas)
        self.d_optimizer = Adam(self.discriminator.parameters(),  # Check parameters
                                lr=self.lr, betas=self.betas)

