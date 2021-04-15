from collections import OrderedDict
import torch.nn as nn

from .vonenet import VOneNet
from .backends.acgan import Discriminator, Generator, weights_init


def get_model(n_classes, channels, latent_dim, image_size=224, **kwargs):
  print('Model: ', 'VOneACGAN')

  discriminator = Discriminator(img_size=image_size, n_classes=n_classes, channels=channels)
  discriminator.apply(weights_init)

  generator = Generator(img_size=image_size, channels=channels,
                      latent_dim=latent_dim, n_classes=n_classes)
  generator.apply(weights_init)

  v_one = VOneNet(image_size=image_size, stride=2, ksize=3, **kwargs)
  bottleneck = nn.Conv2d(512, 16, kernel_size=1, stride=1, bias=False)
  nn.init.kaiming_normal_(bottleneck.weight, mode='fan_out', nonlinearity='relu')

  model = nn.Sequential(OrderedDict([
    ('vone_block', v_one),
    ('bottleneck', bottleneck),
    ('model', discriminator),
  ]))

  nn.DataParallel(model)

  # nn.DataParellel(generator)
  # return discriminator, generator
  return model, generator