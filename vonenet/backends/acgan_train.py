import time

import torch
import torch.nn as nn
from torch.optim import Adam

from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

import numpy as np


class ACGANTrainer(object):

    def __init__(self, discriminator: nn.Module, generator: nn.Module, device: torch.device,
                 num_classes: int, lr: float, b1: float, b2: float, img_size: int,
                 batch_size: int, num_workers: int, latent_dim: int):
        self.name = 'acgan_train'
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.adversarial_loss = nn.BCELoss()
        self.adversarial_loss = self.adversarial_loss.to(device)
        self.auxiliary_loss = nn.CrossEntropyLoss()
        self.auxiliary_loss = self.auxiliary_loss.to(device)
        self.g_optimizer = Adam(self.generator.parameters(),
                                lr=lr, betas=(b1, b2))
        self.d_optimizer = Adam(self.discriminator.parameters(),  # Check parameters
                                lr=lr, betas=(b1, b2))

        self.img_size = img_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.latent_dim = latent_dim

        self.data_loader = self.data()

    def data(self):
        dataset = datasets.MNIST(
            './',
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]))

        data_loader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 num_workers=self.num_workers,
                                 pin_memory=True)

        return data_loader

    def get_sample(self, n_row):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

        # Sample noise
        z = Variable(FloatTensor(np.random.normal(
            0, 1, (n_row ** 2, self.latent_dim))))
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([num for _ in range(n_row) for num in range(n_row)])
        labels = Variable(LongTensor(labels))
        gen_imgs = self.generator(z, labels)
        return gen_imgs

    def __call__(self, inp, target):
        start = time.time()

        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

        target.to(self.device)

        batch_size = inp.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(
            1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0),
                        requires_grad=False)

        # Configure input
        real_inp = Variable(inp.type(FloatTensor))
        labels = Variable(target.type(LongTensor))

        # Train Generator
        self.g_optimizer.zero_grad()

        z = Variable(FloatTensor(np.random.normal(
            0, 1, (batch_size, self.latent_dim))))
        gen_labels = Variable(LongTensor(
            np.random.randint(0, self.num_classes, batch_size)))

        gen_inp = self.generator(z, gen_labels)

        validity, pred_label = self.discriminator(gen_inp)
        g_loss = 0.5 * (self.adversarial_loss(validity, valid) +
                        self.auxiliary_loss(pred_label, gen_labels))

        g_loss.backward()
        self.g_optimizer.step()

        # Train Discriminator
        self.d_optimizer.zero_grad()

        # Loss for real images
        real_pred, real_aux = self.discriminator(real_inp)
        d_real_loss = (self.adversarial_loss(real_pred, valid) +
                       self.auxiliary_loss(real_aux, labels)) / 2

        # Loss for fake images
        fake_pred, fake_aux = self.discriminator(gen_inp.detach())
        d_fake_loss = (self.adversarial_loss(fake_pred, fake) +
                       self.auxiliary_loss(fake_aux, gen_labels)) / 2

        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate(
            [real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate(
            [labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        self.d_optimizer.step()

        record = {}
        record['d_loss'] = d_loss.item()
        record['g_loss'] = g_loss.item()
        record['accuracy'] = 100 * d_acc
        record['duration'] = time.time() - start

        return record
