#Train function for mpl model.
def train_metaModel(s_model, t_model, criterion, t_optimizer, s_optimizer, t_scheduler, s_scheduler, num_epochs=epochs):
  since = time.time()

  #Initialise model.
  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 10)
    
    # Each epoch has a training and validation phase
    for phase in ['training', 'validation']:
      if phase == 'training':
        s_model.train()  # Set model to training mode
        t_model.train()
          
      else:
        s_model.eval()   # Set model to validation mode
        t_model.eval()

      # running_loss = 0.0
      # running_corrects = 0

      # Iterate over data.
      for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(phase == 'training'):
          #Run training for teacher model using original data.
          t_outputs = t_model(inputs)

          #Obtain training predictions.
          soft_preds = torch.softmax(t_outputs, dim = -1) #Generate soft labels
          max_prob, pseudo_preds = torch.max(soft_preds, dim = -1)
          
          #Create dataframe containing image + pseudo-labels
          pseudo_labels = list(pseudo_preds)
          s_inputs = pd.DataFrame(inputs, pseudo_labels)

          # Put dataframe into dataloader object, get inputs and pseudo labels and pass it into gpu device

          #Run training for student model using pseudo-labels. 
          s_preds = s_model(s_inputs)

          #Calculate loss of student model and backward propagate to update parameters.
          s_loss = criterion(s_preds[batch_size:], pseudo_preds)
          s_scheduler.scale(s_loss).backward()
          clip_gradient(s_optimizer, grad_clip) #Clip gradient to prevent exploding gradients
          s_scaler.step(s_optimizer)
          s_scaler.update()
          s_scheduler.step()

        #Evaluate student model.
        with torch.no_grad(phase == 'validation'):
          #Run prediction for student model using validation set.
          s_outputs = s_model(inputs)
          
        #Calculate MPL signal (validation loss) and supervised signal.
        s_val_loss = criterion(s_outputs[batch_size:], labels)
        dot_product = s_loss - s_val_loss
        _, hard_preds = torch.max(unlabelled_images, dim=-1) 
        t_loss = dot_product * criterion(unlabelled_images, hard_preds)

        #Update teacher parameters based on loss (feedback loop)
        t_scaler.scale(t_loss).backward()
        clip_gradient(t_optimizer, grad_clip) #Clip gradient to prevent exploding gradients
        t_scaler.step(t_optimizer)
        t_scaler.update()
        t_scheduler.step()

        #Output training statistics.
        # running_loss += loss.item() * inputs.size(0)
        # running_corrects += torch.sum(hard_preds == labels.data)

      #Calculates epoch statistics.
      # epoch_loss = running_loss / dataset_sizes[phase]
      # epoch_acc = running_corrects.double() / dataset_sizes[phase]

      # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
      #     phase, epoch_loss, epoch_acc))

      #Initialise model if in validation phase and epoch has best accuracy.
      # if phase == 'validation' and epoch_acc > best_acc:
      #     best_acc = epoch_acc
      #     best_model_wts = copy.deepcopy(model.state_dict())

    print()
  
  #Output runtime of model.
  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
      time_elapsed // 60, time_elapsed % 60))
  # print('Best val Acc: {:4f}'.format(best_acc))

  #Save weights of the best epoch model into training directory.
  training_path = f"/content/MetaPseudoModel/training_results/weights/fold{fold}_weights.pth"
  torch.save(model.state_dict(), training_path)

  return model

