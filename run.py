import os, argparse, time, subprocess, io, shlex
import pandas as pd

parser = argparse.ArgumentParser(description='training')


parser.add_argument('--n_epochs', default=600, type=int,
                    help='number of epochs of training')
parser.add_argument('--batch_size', default=128, type=int,
                    help='size of batch for training')
parser.add_argument('--workers', default=2,
                    help='number of data loading workers')
parser.add_argument('--ngpus', default=2, type=int,
                    help='number of GPUs to use; 0 if you want to run on CPU')
parser.add_argument('--img_size', default=64, type=int,
                    help='size of each image dimension')
parser.add_argument('--channels', default=1, type=int,
                    help='number of image channels')
parser.add_argument('--latent_dim', default=128, type=int,
                    help='dimensionality of the latent space')
parser.add_argument('--n_classes', default=10, type=int,
                    help='number of classes for dataset')
parser.add_argument('--lr', default=0.0002, type=float,
                    help='adam: learning rate')
parser.add_argument('--b1', default=0.5, type=float,
                    help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', default=0.999, type=float,
                    help='adam: decay of first order momentum of gradient')
parser.add_argument('--sample_interval', type=int, default=1000,
                    help='interval between image sampling')
parser.add_argument('--model_arch', choices=['acgan', 'dcgan', 'robgan'], default='robgan',
                    help='back-end model architecture to load')

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
        # making sure GPUs are numbered the same way as in nvidia_smi
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            [str(i) for i in gpus['index'].iloc[:n]])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import torch
import torch.nn as nn
from torch.cuda import is_available, device_count
from torchvision.utils import make_grid

from torch.utils.tensorboard import SummaryWriter
import tqdm

from vonenet import get_model
# from vonenet.backends.dcgan_train import DCGANTrainer
from vonenet.backends.robgan_train import RobGANTrainer


set_gpus(FLAGS.ngpus)

device = torch.device('cuda' if is_available() else 'cpu')




def train():
    discriminator, generator = get_model(model_arch=FLAGS.model_arch, image_size=FLAGS.img_size, n_classes=FLAGS.n_classes,
                                         channels=FLAGS.channels, latent_dim=FLAGS.latent_dim)

    if FLAGS.ngpus == 0:
        print('Running on CPU')
    if FLAGS.ngpus > 0 and device_count() > 1:
        print('Running on multiple GPUs')
        generator = nn.DataParallel(generator, list(range(FLAGS.ngpus)))
        discriminator = nn.DataParallel(discriminator, list(range(FLAGS.ngpus)))
        generator = generator.to(device)
        discriminator = discriminator.to(device)
    elif FLAGS.ngpus > 0 and device_count() == 1:
        print('Running on single GPU')
        generator = generator.to(device)
        discriminator = discriminator.to(device)
    else:
        print('No GPU detected!')

    writer = SummaryWriter('logs/vone_robgan_experiment', max_queue=100)
    trainer = RobGANTrainer(discriminator, generator, device, num_classes=FLAGS.n_classes, lr=FLAGS.lr, b1=FLAGS.b1, b2=FLAGS.b2, img_size=FLAGS.img_size,
                           num_workers=FLAGS.workers, batch_size=FLAGS.batch_size, latent_dim=FLAGS.latent_dim)

    start_epoch = 0
    iter_d = 5
    data_loader_iter = trainer.data_loader

    d_losses_fake = 0.0
    d_losses_real = 0.0
    g_losses = 0.0
    # accs_r = 0.0
    accs_r1 = 0.0
    # accs_f  = 0.0
    accs_f1  = 0.0

    start = time.time()
    for epoch in tqdm.trange(start_epoch, FLAGS.n_epochs + 1, initial=0, desc='epoch'):
        for idx, data in enumerate(tqdm.tqdm(data_loader_iter, desc=trainer.name)):
            with_gen = (idx % iter_d) == 0
            record = trainer(with_gen, *data)

            d_losses_fake += record['d_loss_fake'] 
            d_losses_real += record['d_loss_real']
            if with_gen:
                g_losses += record['g_loss']
            # accs_r += record['acc_r'] 
            accs_r1 += record['acc_r@1']
            # accs_f += record['acc_f'] 
            accs_f1 += record['acc_f@1'] 

            batches_done = epoch * len(data_loader_iter) + idx

            if batches_done % FLAGS.sample_interval == 0:
                writer.add_scalar('train/loss/discriminator',
                                  (d_losses_fake+d_losses_real) / FLAGS.sample_interval, batches_done)
                writer.add_scalar('train/loss_d_real',
                                  d_losses_real / FLAGS.sample_interval, batches_done)
                writer.add_scalar('train/loss_d_fake',
                                  d_losses_fake / FLAGS.sample_interval, batches_done)
                writer.add_scalar('train/loss/generator',
                                  g_losses / FLAGS.sample_interval, batches_done)
                writer.add_scalar('train/acc_r1',
                                  accs_r1 / FLAGS.sample_interval, batches_done)
                # writer.add_scalar('train/acc_r',
                #                   accs_r / FLAGS.sample_interval, batches_done)
                writer.add_scalar('train/acc_f1',
                                  accs_f1 / FLAGS.sample_interval, batches_done)
                # writer.add_scalar('train/acc_f',
                #                   accs_f / FLAGS.sample_interval, batches_done)
                writer.flush()

                d_losses_fake = 0.0
                d_losses_real = 0.0
                g_losses = 0.0
                # accs_r = 0.0
                accs_r1 = 0.0
                # accs_f  = 0.0
                accs_f1  = 0.0

        writer.add_image(
            'train/samples', make_grid(trainer.get_sample(), normalize=True), epoch)
        # if epoch % 1 == 0:# save model
        #         torch.save(trainer.discriminator.state_dict(), f'./assets/models/dis_epoch_{epoch}.pth')
        #         torch.save(trainer.generator.state_dict(), f'./assets/models//gen_epoch_{epoch}.pth')
        #         torch.save(trainer.d_optimizer.state_dict(), f'./assets/models/d_opt_epoch_{epoch}.pth')
        #         torch.save(trainer.g_optimizer.state_dict(), f'./assets/models//g_opt_epoch_{epoch}.pth')
                
        if (epoch + 1) % 50 == 0:# change step size
            trainer.update_optimizers()
        duration = (time.time() - start) / len(data_loader_iter)
        print('[Epoch %d/%d] [Duration: %d]' %
              (epoch, FLAGS.n_epochs, duration))


if __name__ == '__main__':
    train()
