import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = "thin_cloud/training"
VAL_DIR = "thin_cloud/validation"

LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
BETA_ONE = 0.5
BETA_TWO = 0.999
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = False

CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"
