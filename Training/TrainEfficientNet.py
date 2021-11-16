#Fold code for EfficientNet training.
for fold, (train_idx, valid_idx) in enumerate(strat_kfold.split(X, y)):
    fold_no = fold+1
    print(f"FOLD {fold_no}") # Keep track of fold number 

    #Convert training and validation data to custom class.
    train_df = trainLabel_df.iloc[train_idx][["Patient_no", "Patient_ID", "labels"]]
    valid_df = trainLabel_df.iloc[valid_idx][["Patient_no", "Patient_ID", "labels"]]
    train_set = CellsDataset(train_df, train_images)
    valid_set = CellsDataset(valid_df, train_images)

    #Load training and validation sets to dataloader.
    valid_loader = DataLoader(valid_set, batch_size=batch_size, 
            num_workers=num_workers, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, 
            num_workers=num_workers, shuffle=True)
    dataloaders = {'training': train_loader, 'validation': valid_loader}
    dataset_sizes = {'training': len(train_set), 'validation': len(valid_set)}
    class_names = [0, 1] #Binary class labels.
    
    #Train EfficientNet model.
    efficientNet_training(base_model, criterion, optimizer, scheduler, 
            num_epochs = epochs, fold = num_fold)
