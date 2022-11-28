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
    loop = tqdm(train_dataloader, leave=True)

    for cur_index, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Discriminator Training
        with torch.cuda.amp.autocast():
            y_fake = generator(x)
            discriminator_real = discriminator(x, y)
            discriminator_fake = discriminator(x, y_fake.detach())  # Important to detach else, computation breaks

            discriminator_real_loss = loss_bce(discriminator_real, torch.ones_like(discriminator_real))
            discriminator_fake_loss = loss_bce(discriminator_fake, torch.zeros_like(discriminator_fake))

            # As per paper, it is divided by 2 to delay the discriminator_training process compared to generator
            discriminator_total_loss = (discriminator_real_loss + discriminator_fake_loss) / 2

        # Updating discriminator values
        discriminator.zero_grad()
        scalar_discriminator.scale(discriminator_total_loss).backward()
        scalar_discriminator.step(optimizer_discriminator)
        scalar_discriminator.update()

        # Generator Training
        with torch.cuda.amp.autocast():
            discriminator_fake = discriminator(x, y_fake)
            generator_fake_loss = loss_bce(discriminator_fake, torch.ones_like(discriminator_fake))
            generator_loss_l1 = loss_l1(y_fake, y) * config.L1_LAMBDA
            generator_total_loss = generator_fake_loss + generator_loss_l1

        # Updating generator values
        optimizer_generator.zero_grad()
        scalar_generator.scale(generator_total_loss).backward()
        scalar_generator.step(optimizer_generator)
        scalar_generator.update()

        if cur_index % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(discriminator_real).mean().item(),
                D_fake=torch.sigmoid(discriminator_fake).mean().item()
            )


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
