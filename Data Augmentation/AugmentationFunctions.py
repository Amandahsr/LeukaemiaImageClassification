#Function for augmentation details.
def augmentation(img):
    img = torch.from_numpy(img).type(torch.DoubleTensor) # Change dtype from numpy to tensor.
    img = torch.moveaxis(img, -1, 0)

    affine_transformer = transforms.RandomAffine(degrees=(0, 360), # Rotates image.
                                                 translate=(0.1, 0.1), #Offsets X/Y axes.
                                                 scale=(0.95, 1.05)) #Scales size.

    affine_img = affine_transformer(img).type(torch.LongTensor)
    affine_img = torch.moveaxis(affine_img, 0, -1)

    return affine_img

#Augmentation function.
"""
    #Saves augmented image into op_dir given ip_dir of images and csv file/df.
    :param rounds: no.of augmented images generated per image.
    :param ip_dir: directory of image files.
    :param op_dir: directory for augmented images.
    :param csv: labels.csv.
    :param df: pandas df for labels.
    :param idx: index generated from train-test split.
    :return: saves temp_aug directory + aug_labels df.
"""

def augment(rounds, ip_dir="../Data_main/images/", op_dir="../Data_main/temp_aug/", labels=None):
    
    #Initialise input and output directory.
    IP_DIR = ip_dir
    OP_DIR = op_dir

    #Create augmentation directory.
    if not os.path.exists(OP_DIR):
        os.makedirs(OP_DIR)

    #Table to read images for augmentation.
    try:
        labels_df = pd.read_csv(labels) # For csv inputs.
    except:
        labels_df = labels # For df inputs.

    #Create dataframe to attach labels to augmented images.
    aug_l = []

    #Iterate through labels_df to augment images.
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

            #Add augmented image info to dataframe.
            aug_l.append([patient_no, aug_name, label])
            #Save augmented image in bmp format.
            plt.imsave('{}{}_{}.bmp'.format(OP_DIR, i, id), augmented)

    aug_df = pd.DataFrame(aug_l, columns=["Patient_no", "Patient_ID", "labels"])

    return aug_df

# TESTCASE #
# aug_df = augment(rounds=5, op_dir="../test_augment/", labels="../Data_main/train_labels.csv")
# print(aug_df)
