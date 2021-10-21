#K-fold cross validation code that trains base EfficientNet model.
#Data augmentation is removed until we gauge how much space we have.
for fold, (train_idx, valid_idx) in enumerate(strat_kfold.split(X, y)):
    print("FOLD {}".format(fold))

    train_df = label_df.iloc[train_idx][["Patient_no", "Patient_ID", "labels"]]
    valid_df = label_df.iloc[valid_idx][["Patient_no", "Patient_ID", "labels"]]
    train_set = CellsDataset(train_df, "/content/ZB4171_LeukemiaImageClassification-Ongoing-/Data_main/images")
    valid_set = CellsDataset(valid_df, "/content/ZB4171_LeukemiaImageClassification-Ongoing-/Data_main/images")

    #Add augmentation data to original training set.
    # aug_dir = "../test_augment/"
    # aug_df = augment(rounds=aug_rounds, op_dir=aug_dir, labels=train_df)
    # aug_set = CellsDataset(aug_df, aug_dir)
    # train_aug_set = torch.utils.data.ConcatDataset([train_set, aug_set])

    #Dataloader for input into the model (need to integrate into correct format for efficientnet).
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    # input takes in dictionary format
    # this part is combining our CellsDataset class with the input format for effiecientnet
    dataloaders = {'training': train_loader, 'validation': valid_loader}
    dataset_sizes = {'training': len(train_set), 'validation': len(valid_set)}
    class_names = [0, 1] # hard code labels based on labels in csv
    
    # train data on n epochs
    train_model(base_model, criterion, optimizer, lr_scheduler, num_epochs=epochs)

    #Delete augmented training directory before next training fold to save disk space.
    #shutil.rmtree(aug_dir)
