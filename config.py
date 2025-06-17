import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PREFIX = "road"
EPOCHS = 10
SAVE_MODEL_PATH = "saved_models"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DATA_PATH = "Road_Sign_Detection"
NUM_CLASSES = 4
BATCH_SIZE = 4
