import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

def augmentation(img):
    img = torch.from_numpy(img).type(torch.DoubleTensor)
    img = torch.moveaxis(img, -1, 0)
    affine_transformer = transforms.RandomAffine(degrees=(0, 360), translate=(0.1, 0.1), scale=(0.95, 1.05))
    affine_img = affine_transformer(img).type(torch.LongTensor)
    affine_img = torch.moveaxis(affine_img, 0, -1)
    return affine_img

def main(rounds):
    IP_DIR = "C:/Users/Tiana/Documents/ZB4171/ALL Classification/eg_data/" # directory of all image files
    OP_DIR = "C:/Users/Tiana/Documents/ZB4171/ALL Classification/test_augment/" # directory of augmented images

    if not os.path.exists(OP_DIR):
        os.makedirs(OP_DIR)

    file_list = os.listdir(IP_DIR)
    for file in tqdm(file_list):

        if not 'bmp' in file:
            continue

        id = file.split('.')[0] # get name w/o .bmp extension
        img = plt.imread(IP_DIR + file)

        for i in range(rounds):
            augmented = augmentation(img).detach().numpy().astype('uint8')
            plt.imsave('{}{}_{}.bmp'.format(OP_DIR, id, i), augmented) # save image in bmp format

if __name__=='__main__':
    main(rounds=5)