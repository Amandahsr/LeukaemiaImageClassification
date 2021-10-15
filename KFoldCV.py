from DataSet import CellsDataset

#Initialise labels file.
label_csv = "../Data_main/train_labels.csv"

# Define kfold cross validator
strat_kfold = StratifiedKFold(n_splits=no_fold, shuffle=True, random_state=42)

# Initialise dataset object
# dataset = CellsDataset("../Data_main/train_labels.csv", "./Data_main/images") # to fill in arguments

# Getting indexes for stratified k fold
label_df = pd.read_csv(label_csv)
X = np.zeros(len(label_df))
y = label_df['labels']

# Preparing data for training on each fold
for fold, (train_idx, valid_idx) in enumerate(strat_kfold.split(X, y)):
    print("FOLD {}".format(fold))

    train_df = label_df.iloc[train_idx][["Patient_no", "Patient_ID", "labels"]]
    valid_df = label_df.iloc[valid_idx][["Patient_no", "Patient_ID", "labels"]]
    train_set = CellsDataset(train_df, "./Data_main/images")
    valid_set = CellsDataset(valid_df, "./Data_main/images")

    # Add in augmentation data on training set
    aug_dir = "../test_augment/"
    aug_df = augment(rounds=5, op_dir=aug_dir, labels=train_df)
    aug_set = CellsDataset(aug_df, aug_dir)
    # concat aug_df with train_df
    train_aug_set = torch.utils.data.ConcatDataset([train_set, aug_set])

    # dataloader for input into the model (need to integrate into correct format for efficientnet)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_aug_set, batch_size=batch_size, shuffle=True)
    # input takes in dictionary format
    dataloaders = {'training': train_loader,'validation': valid_loader}

    """
    - put efficientnet model here
    - do training here
    - evaluate loss, accuracy here
    """
    # delete the augmented training dir bef training on next fold to save disk space
    shutil.rmtree(aug_dir)
