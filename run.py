import os, argparse, time, subprocess, io, shlex
import pandas as pd

parser = argparse.ArgumentParser(description='training')


parser.add_argument('--n_epochs', default=200, type=int,
                    help='number of epochs of training')
parser.add_argument('--batch_size', default=128, type=int,
                    help='size of batch for training')
parser.add_argument('--workers', default=2,
                    help='number of data loading workers')
parser.add_argument('--ngpus', default=0, type=int,
                    help='number of GPUs to use; 0 if you want to run on CPU')
parser.add_argument('--img_size', default=32, type=int,
                    help='size of each image dimension')
parser.add_argument('--channels', default=3, type=int,
                    help='number of image channels')
parser.add_argument('--latent_dim', default=100, type=int,
                    help='dimensionality of the latent space')
parser.add_argument('--n_classes', default=10, type=int,
                    help='number of classes for dataset')
parser.add_argument('--lr', default=0.0002, type=float,
                    help='adam: learning rate')
parser.add_argument('--b1', default=0.5, type=float,
                    help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', default=0.999, type=float,
                    help='adam: decay of first order momentum of gradient')

FLAGS, FIRE_FLAGS = parser.parse_known_args()

def set_gpus(n=2):
    """
    Finds all GPUs on the system and restricts to n of them that have the most
    free memory.
    """
    if n > 0:
        gpus = subprocess.run(shlex.split(
            'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits'), check=True,
            stdout=subprocess.PIPE).stdout
        gpus = pd.read_csv(io.BytesIO(gpus), sep=', ', engine='python')
        gpus = gpus[gpus['memory.total [MiB]'] > 10000]  # only above 10 GB
        if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
            visible = [int(i)
                       for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
            gpus = gpus[gpus['index'].isin(visible)]
        gpus = gpus.sort_values(by='memory.free [MiB]', ascending=False)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # making sure GPUs are numbered the same way as in nvidia_smi
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            [str(i) for i in gpus['index'].iloc[:n]])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

set_gpus(FLAGS.ngpus)

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import tqdm
import numpy as np
from vonenet import get_model
from vonenet.backends.acgan import Generator, weights_init

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if cuda else 'cpu')


def train():
    discriminator = get_model(image_size=FLAGS.img_size, n_classes=FLAGS.n_classes,
                              channels=FLAGS.channels)

    generator = Generator(img_size=FLAGS.img_size, channels=FLAGS.channels,
                          latent_dim=FLAGS.latent_dim, n_classes=FLAGS.n_classes)

    if FLAGS.ngpus == 0:
        print('Running on CPU')
    if FLAGS.ngpus > 0 and torch.cuda.device_count() > 1:
        print('Running on multiple GPUs')
        generator = generator.to(device)
        discriminator = discriminator.to(device)
    elif FLAGS.ngpus > 0 and torch.cuda.device_count() == 1:
        print('Running on single GPU')
        generator = generator.to(device)
        discriminator = discriminator.to(device)
    else:
        print('No GPU detected!')
        # discriminator = discriminator.module

    trainer = VOneNetTrainer(discriminator, generator)
    trainer()
    return

class VOneNetTrainer(object):

    def __init__(self, discriminator, generator):
        self.name = 'train'
        self.generator = generator
        self.discriminator = discriminator
        self.adversarial_loss = nn.BCELoss()
        self.adversarial_loss = self.adversarial_loss.to(device)
        self.auxiliary_loss = nn.CrossEntropyLoss()
        self.auxiliary_loss = self.auxiliary_loss.to(device)
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(),
                                            lr=FLAGS.lr, betas=(FLAGS.b1, FLAGS.b2))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), # Check parameters
                                            lr=FLAGS.lr, betas=(FLAGS.b1, FLAGS.b2))
        self.data_loader = self.data()

    def data(self):
        dataset = datasets.CIFAR10(
                '../../data/cifar10',
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.Resize(FLAGS.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                ]))

        data_loader = DataLoader(dataset,
                                batch_size=FLAGS.batch_size,
                                shuffle=True,
                                num_workers=FLAGS.workers,
                                pin_memory=True)

        return data_loader

    def sample_image(self, n_row, batches_done):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

        # Sample noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, FLAGS.latent_dim))))
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([num for _ in range(n_row) for num in range(n_row)])
        labels = Variable(LongTensor(labels))
        gen_imgs = self.generator(z, labels)
        save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)

    def __call__(self):
        self.generator.apply(weights_init)

        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

        start = time.time()
        for epoch in range(FLAGS.n_epochs):
            for idx, (inp, target) in enumerate(tqdm.tqdm(self.data_loader, desc=self.name)):
                target.to(device)

                batch_size = inp.shape[0]

                # Adversarial ground truths
                valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_inp = Variable(inp.type(FloatTensor))
                labels = Variable(target.type(LongTensor))


                # Train Generator
                self.g_optimizer.zero_grad()

                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, FLAGS.latent_dim))))
                gen_labels = Variable(LongTensor(np.random.randint(0, FLAGS.n_classes, batch_size)))

                gen_inp = self.generator(z, gen_labels)

                validity, pred_label = self.discriminator(gen_inp)
                g_loss = 0.5 * (self.adversarial_loss(validity, valid) + self.auxiliary_loss(pred_label, gen_labels))

                g_loss.backward()
                self.g_optimizer.step()

                # Train Discrriminator
                self.d_optimizer.zero_grad()

                # Loss for real images
                real_pred, real_aux = self.discriminator(real_inp)
                d_real_loss = (self.adversarial_loss(real_pred, valid) + self.auxiliary_loss(real_aux, labels)) / 2

                # Loss for fake images
                fake_pred, fake_aux = self.discriminator(gen_inp.detach())
                d_fake_loss = (self.adversarial_loss(fake_pred, fake) + self.auxiliary_loss(fake_aux, gen_labels)) / 2

                d_loss = (d_real_loss + d_fake_loss) / 2

                # Calculate discriminator accuracy
                pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
                gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
                d_acc = np.mean(np.argmax(pred, axis=1) == gt)

                d_loss.backward()
                self.d_optimizer.step()

                print(
                    '[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]'
                    % (epoch, FLAGS.n_epochs, idx, len(self.data_loader), d_loss.item(), 100 * d_acc, g_loss.item())
                )
                batches_done = epoch * len(self.data_loader) + idx
                self.sample_image(n_row=10, batches_done=batches_done)

            duration = (time.time() - start) / len(self.data_loader)
            print('[Epoch %d/%d] [Duration: %d]' % (epoch, FLAGS.n_epochs, duration))

if __name__ == '__main__':
    train()