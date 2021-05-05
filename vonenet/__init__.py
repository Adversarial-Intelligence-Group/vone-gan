from collections import OrderedDict
import torch.nn as nn

from .vonenet import VOneNet
from .backends import acgan, dcgan, weights_init, robgan


def get_model(model_arch, n_classes, channels, latent_dim, image_size=224, **kwargs):

    if model_arch.lower() == 'acgan':
        print('Model: acgan')
        discriminator = acgan.Discriminator(
            img_size=image_size, n_classes=n_classes, channels=channels)
        generator = acgan.Generator(
            img_size=image_size, channels=channels, latent_dim=latent_dim, n_classes=n_classes)
    elif model_arch.lower() == 'dcgan':
        print('Model: dcgan')
        discriminator = dcgan.Discriminator()
        generator = dcgan.Generator(channels=channels)
    elif model_arch.lower() == 'robgan':
        print('Model: robgan')
        discriminator = robgan.Discriminator(ch=16, n_classes=n_classes)
        generator = robgan.Generator(out_ch=channels,ch=16, dim_z=latent_dim, n_classes=n_classes, bottom_width=4) # just 4
    else:
        raise NotImplementedError('Model not implemented')

    if model_arch.lower() != 'robgan':
        discriminator.apply(weights_init)
        generator.apply(weights_init)

    v_one = VOneNet(image_size=image_size, complex_channels=256, simple_channels=256, stride=2, ksize=3, **kwargs)
    # v_one = VOneNet(image_size=image_size, complex_channels=128, simple_channels=128, stride=2, ksize=3, **kwargs)

    
    if model_arch.lower() == 'robgan':
        bottleneck = nn.Conv2d(512, 32, kernel_size=1, stride=1, bias=False)
    else:
        bottleneck = nn.Conv2d(512, 16, kernel_size=1, stride=1, bias=False)
    nn.init.kaiming_normal_(
        bottleneck.weight, mode='fan_out', nonlinearity='relu')

    model = nn.Sequential(OrderedDict([
        ('vone_block', v_one),
        ('bottleneck', bottleneck),
        ('model', discriminator),
    ]))

    return model, generator
