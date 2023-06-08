import torch

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
dataset_path = './imdb-dataset.csv'
MAX_LEN=600
BATCH_SIZE = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 2
EMBED_DIM = 256
OUTPUT_SIZE = 1
DROP_OUT_P=0.5
LR = 0.0005
EPOCHS = 5