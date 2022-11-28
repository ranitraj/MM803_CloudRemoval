import torch
import torch.nn as nn
import torch.optim as optim
import config

from data_mapper import DataMapper
from model_generator import Generator
from model_discriminator import Discriminator
from utils import load_checkpoint

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

    # Loading configuration for Discriminator and Generator
    if config.FLAG_LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GENERATOR, generator, optimizer_generator, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISCRIMINATOR, discriminator, optimizer_discriminator, config.LEARNING_RATE,
        )

    # Training dataset
    train_dataset = DataMapper(root_dir=config.DIRECTORY_TRAINING)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.COUNT_WORKERS,
    )

    # Adding scalars for performing efficient steps while gradient scaling
    scalar_discriminator = torch.cuda.amp.GradScaler()
    scalar_generator = torch.cuda.amp.GradScaler()

if __name__ == "__main__":
    main()
