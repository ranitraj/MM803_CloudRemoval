import torch
import torch.nn as nn
import torch.optim as optim
import config

from model_generator import Generator
from model_discriminator import Discriminator

torch.backends.cudnn.benchmark = True


def main():
    # Initializing discriminator and generator
    discriminator = Discriminator(in_channels=3).to(config.DEVICE)
    generator = Generator(in_channels=3, features=64).to(config.DEVICE)

    # Initializing optimizers for discriminator and generator
    optimizer_discriminator = optim.Adam(
        discriminator.parameters(),
        lr=config.LEARNING_RATE,
        betas=(config.BETA_ONE,
               config.BETA_TWO)
    )
    optimizer_generator = optim.Adam(
        generator.parameters(),
        lr=config.LEARNING_RATE,
        betas=(config.BETA_ONE,
               config.BETA_TWO)
    )

    # Initializing the two Loss functions for Pix2Pix GAN
    loss_bce = nn.BCEWithLogitsLoss()
    loss_l1 = nn.L1Loss()  # Works well with PatchGAN


if __name__ == "__main__":
    main()
