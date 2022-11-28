import torch
import torch.nn as nn
import torch.optim as optim
import config

from data_mapper import DataMapper
from model_generator import Generator
from model_discriminator import Discriminator
from utils import load_checkpoint, save_checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.backends.cudnn.benchmark = True


def start_training_dataset(discriminator, generator, train_dataloader, optimizer_discriminator,
                           optimizer_generator, loss_l1, loss_bce, scalar_generator, scalar_discriminator):
    pass


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

    # Starts loading checkpoint for Discriminator and Generator if FLAG_LOAD_MODEL is set to 'True' in config.py file
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

    # Starts training Generator and Discriminator using the training dataset
    for cur_epoch in range(config.NUM_EPOCHS):
        start_training_dataset(
            discriminator, generator, train_dataloader, optimizer_discriminator,
            optimizer_generator, loss_l1, loss_bce, scalar_generator, scalar_discriminator,
        )

        # Starts saving checkpoints if FLAG_SAVE_MODEL is set to 'True' in config.py file
        if config.FLAG_SAVE_MODEL and cur_epoch % 5 == 0:
            save_checkpoint(generator, optimizer_generator, filename=config.CHECKPOINT_GENERATOR)
            save_checkpoint(discriminator, optimizer_discriminator, filename=config.CHECKPOINT_DISCRIMINATOR)


if __name__ == "__main__":
    main()
