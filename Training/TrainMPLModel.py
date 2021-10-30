#Prepare data for training on each fold for mpl model.
for fold, (train_idx, valid_idx) in enumerate(strat_kfold.split(X, y)):
    print("FOLD {}".format(fold+1))
    
    #Split training set into labelled and unlabelled set: labelled(40%)/unlabelled(60%)
    labelled_idx = []
    unlabelled_idx = []
    for idx in train_idx:
      index = train_idx[idx]
      
      if (index % 5 == 0) or (index % 5 == 1):
        labelled_idx.append(index)
      else:
        unlabelled_idx.apped(index)
    
    labelled_df = label_df.iloc[labelled_idx][["Patient_no", "Patient_ID", "labels"]]
    unlabelled_df = label_df.iloc[unlabelled_idx][["Patient_no", "Patient_ID", "labels"]]
    valid_df = label_df.iloc[valid_idx][["Patient_no", "Patient_ID", "labels"]]

    labelled_set = CellsDataset(labelled_df, "/content/ZB4171_LeukemiaImageClassification-Ongoing-/Data_main/images")
    unlabelled_set = CellsDataset(unlabelled_df, "/content/ZB4171_LeukemiaImageClassification-Ongoing-/Data_main/images")
    valid_set = CellsDataset(valid_df, "/content/ZB4171_LeukemiaImageClassification-Ongoing-/Data_main/images")

    # #Add augmentation data to original training set.
    # aug_dir = "../test_augment/"
    # aug_df = augment(rounds=aug_rounds, op_dir=aug_dir, labels=train_df)
    # aug_set = CellsDataset(aug_df, aug_dir)
    # train_aug_set = torch.utils.data.ConcatDataset([train_set, aug_set])

    #Dataloader for input into the model (need to integrate into correct format for efficientnet).
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    labeltrng_loader = torch.utils.data.DataLoader(labelled_set, batch_size=batch_size, shuffle=True)
    unlabeltrng_loader = torch.utils.data.DataLoader(unlabelled_set, batch_size=batch_size, shuffle=True)

    # input takes in dictionary format
    # this part is combining our CellsDataset class with the input format for effiecientnet
    dataloaders = {'training_lab': valid_loader, 'training_unlab': train_loader, 'validation': valid_loader}
    dataset_sizes = {'training': len(train_set), 'validation': len(valid_set)}
    class_names = [0, 1] # hard code labels based on labels in csv
    
    # train data on n epochs
    trained_mplModel = train_metaModel(s_model, t_model, criterion, t_optimizer, s_optimizer, t_scheduler, s_scheduler, num_epochs=epochs)

    #Delete augmented training directory before next training fold to save disk space.
    #shutil.rmtree(aug_dir)
