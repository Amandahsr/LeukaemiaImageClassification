#This script initialises data and parameters required for training and testing of model.

pip install efficientnet_pytorch

from efficientnet_pytorch import EfficientNet
import json
from PIL import Image
import torch
from torchvision import transforms

#Initialise parameters required for training model.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Use CUDA to run training, else use CPU.
criterion = nn.CrossEntropyLoss() #Loss function.
learning_rate = 0.001
momentum = 0.9

#Normalises images according to pretrained EfficientNet-B0 values
transform_data = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

#Data directory
data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          transform_data[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
