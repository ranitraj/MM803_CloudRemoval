import torch
import config


def load_checkpoint(type_checkpoint_file, type_model, type_optimizer, learning_rate):
    """
    Loads the checkpoint to maintain progress
    
    :param type_checkpoint_file: generator/ discriminator checkpoint file
    :param type_model: generator/ discriminator model
    :param type_optimizer: optimizer used for generator/ discriminator
    :param learning_rate: learning rate supplied
    """
    print(f"Loading checkpoint: File = {type_checkpoint_file}, Model = {type_model} & Optimizer = {type_optimizer}")
    checkpoint = torch.load(type_checkpoint_file, map_location=config.DEVICE)
    type_model.load_state_dict(checkpoint["state_dict"])
    type_optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    for param_group in type_optimizer.param_groups:
        param_group["lr"] = learning_rate
