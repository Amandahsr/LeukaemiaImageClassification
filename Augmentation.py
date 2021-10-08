import os
import torch
import pandas as pd
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

def augment(rounds, ip_dir="../Data_main/images/", op_dir="../Data_main/temp_aug/", labels=None):
    """
    # Saves augmented image into op_dir given ip_dir of images and csv file/df.
    :param rounds: no.of augmented images generated per image
    :param ip_dir: directory of image files
    :param op_dir: directory for augmented images
    :param csv: labels.csv
    :param df: pandas df for labels
    :param idx: index generated from train-test split
    :return: saves temp_aug directory + aug_labels df
    """
    IP_DIR = ip_dir
    OP_DIR = op_dir

    # get table to read images to augment
    try:
        # if input is csv
        labels_df = pd.read_csv(labels)
    except:
        # if input is df
        labels_df = labels

    # Make augmentation directory
    if not os.path.exists(OP_DIR):
        os.makedirs(OP_DIR)

    # create new aug dataframe: attach labels to aug imgs
    aug_l = []

    # iterate through all rows in labels_df to augment imgs
    for idx, row in tqdm(labels_df.iterrows()):
        bmp = row["Patient_ID"]
        id = bmp.split('.')[0] # get name w/o .bmp extension
        label = row["labels"]
        patient_no = row["Patient_no"]

        try:
            img = plt.imread(IP_DIR + bmp)
        except:
            print('{} not in image folder'.format(bmp))
            continue

        for i in range(rounds):
            augmented = augmentation(img).detach().numpy().astype('uint8')
            aug_name = "{}_{}".format(i, id)

            # add new data to dataframe
            aug_l.append([patient_no, aug_name, label])
            # save image in bmp format
            plt.imsave('{}{}_{}.bmp'.format(OP_DIR, i, id), augmented)

    aug_df = pd.DataFrame(aug_l, columns=["Patient_no", "Patient_ID", "labels"])
    return aug_df

# TESTCASE #
# aug_df = augment(rounds=5, op_dir="../test_augment/", labels="../Data_main/train_labels.csv")
# print(aug_df)
