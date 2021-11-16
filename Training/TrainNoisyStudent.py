#Fold code for Noisy Student training.
for fold, (train_idx, valid_idx) in enumerate(strat_kfold.split(X, y)):
    fold_no = fold+1
    print("FOLD {}".format(fold_no+1))
    
    #Split training set into labelled(50%) & unlabelled(50%) sets.
    labelled_idx = []
    unlabelled_idx = []

    for idx in range(len(train_idx)):
      split_idx = train_idx[idx]
      if (split_idx % 2 == 0):
        labelled_idx.append(split_idx)
      else:
        unlabelled_idx.append(split_idx)
    
    #Convert labelled, unlabelled and validation sets to custom class.
    labelled_df = trainLabel_df.iloc[labelled_idx][["Patient_no", "Patient_ID", "labels"]]
    unlabelled_df = trainLabel_df.iloc[unlabelled_idx][["Patient_no", "Patient_ID", "labels"]]
    pseudolabel_df = unlabelled_df[["Patient_no", "Patient_ID"]]
    valid_df = trainLabel_df.iloc[valid_idx][["Patient_no", "Patient_ID", "labels"]]
    valid_set = CellsDataset(valid_df, train_images)
    labelled_set = CellsDataset(labelled_df, train_images)
    unlabelled_set = CellsDataset(unlabelled_df, train_images)

    #Load labelled, unlabelled and validation sets to dataloader.
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=num_workers, 
                              pin_memory=True, shuffle=True)
    label_loader = DataLoader(labelled_set, batch_size=batch_size, num_workers=num_workers, 
                              pin_memory=True, shuffle=True)
    unlabel_loader = DataLoader(unlabelled_set, batch_size=batch_size, num_workers=num_workers, 
                                pin_memory=True, shuffle=True)
    label_dataloader = {'training': label_loader, 'validation': valid_loader}
    dataset_sizes_l = {'training': len(labelled_set), 'validation': len(valid_set)}
    class_names = [0, 1] #Binary class labels.

    #Noisy student loop.
    NS_iterations = 3
    for iteration in range(NS_iterations):
      if iteration == 0:
        print("TRAINING TEACHER:")
        noisyStudentTraining(base_model, label_dataloader, dataset_sizes_l, 
                             criterion, optimizer, scheduler, 
                             num_epochs = epochs, fold=num_fold) #Train first teacher model on labelled images.

      else:
        base_model.eval()

        #Predict pseudolabels from unlabelled images.
        infer_pseudolabels = []
        for inputs, _ in unlabel_loader:
            inputs = inputs.to(device)
            with torch.no_grad():
              output = base_model(inputs) #Teacher model prediction.
              soft_preds = torch.softmax(output, dim = -1) #Soft labels.
              max_prob, hard_preds = torch.max(soft_preds, dim = -1) #Hard labels.
              hard_preds = hard_preds.cpu().detach().numpy()
              infer_pseudolabels.extend(hard_preds)

        #Attach pseudolabels with unlabelled images and send to dataloader.
        pseudolabel_df['labels'] = infer_pseudolabels
        pseudolabel_set = CellsDataset(pseudolabel_df, unlabelled_images)
        pseudolabel_loader = torch.utils.data.DataLoader(pseudolabel_set, batch_size=batch_size, shuffle=True)
        pseudo_dataloader = {'training': pseudolabel_loader, 'validation': valid_loader}
        dataset_sizes_u = {'training': len(pseudolabel_set), 'validation': len(valid_set)}

        print(f"TRAINING STUDENT ITERATION: {iteration}")
        noisyStudentTraining(base_model, pseudo_dataloader, dataset_sizes_u, 
                             criterion, optimizer, scheduler, 
                             num_epochs = epochs, fold = num_fold) #Train student model on labelled images.

