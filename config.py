import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DIRECTORY_TRAINING = "thin_cloud/training/"
DIRECTORY_VALIDATION = "thin_cloud/validation/"

LEARNING_RATE = 2e-4
BATCH_SIZE = 16
COUNT_WORKERS = 2
IMAGE_SIZE = 256
IMAGE_CHANNELS = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
BETA_ONE = 0.5
BETA_TWO = 0.999
NUM_EPOCHS = 500

FLAG_LOAD_MODEL = False
FLAG_SAVE_MODEL = False

CHECKPOINT_DISCRIMINATOR = "disc.pth.tar"
CHECKPOINT_GENERATOR = "gen.pth.tar"
