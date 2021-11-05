def train_metaModel(s_model, t_model, criterion, t_optimizer, s_optimizer, t_scheduler, s_scheduler, num_epochs=epochs):
  since = time.time()

  # Initialise model.
  best_model_wts = copy.deepcopy(s_model.state_dict())
  best_acc = 0.0
  best_epoch = 0

  for epoch in range(num_epochs):
    epoch_start = time.time()
    epoch_no = epoch + 1
    print('Epoch {}/{}'.format(epoch_no, num_epochs))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['training', 'validation']:
      # Training phase
      if phase == 'training':
        s_model.train()  # Set model to training mode
        t_model.train()

        # iterate through size of labelled and unlabelled set
        for i in range(len(labelled_set)):
          # load labelled and unlabelled dataloaders
          labelled_loader = dataloaders['training'][
            0]  # to put training dataloaders as a list in dataloaders dict
          unlabelled_loader = dataloaders['training'][1]

          # Labelled dataloader
          labelled_iter = iter(labelled_loader)
          images_l, targets = labelled_iter.next()
          images_l, targets = images_l.to(device), targets.to(device)
          batch_size = images_l.shape[0]

          # Unlabelled dataloader
          unlabelled_iter = iter(unlabelled_loader)
          images_u, _ = unlabelled_iter.next()
          images_u = images_u.to(device)

          # Initialise images for teacher and student
          all_images = torch.cat((images_l, images_u))

          # Set parameter gradients to 0.
          s_optimizer.zero_grad()
          t_optimizer.zero_grad()

          with torch.set_grad_enabled(phase == 'training'):
            # Run initial training for teacher model
            t_outputs = t_model(all_images)
            # Split t_outputs based on labelled and unlabelled
            t_outputs_l = t_outputs[:batch_size]
            t_outputs_u = t_outputs[batch_size:]

            # Obtain training predictions on unlabelled
            soft_preds = torch.softmax(t_outputs_u, dim=-1)
            max_prob, pseudo_preds = torch.max(soft_preds, dim=-1)

            # Obtain teacher's loss on labelled
            t_loss_l = criterion(t_outputs_l, targets)

            # Run training for student model
            s_outputs = s_model(all_images)
            # Split s_outputs based on labelled and unlabelled
            s_outputs_l = s_outputs[:batch_size]
            s_outputs_u = s_outputs[batch_size:]

            # Obtain loss on labelled
            s_loss_l_old = criterion(s_outputs_l, targets)

            # Obtain loss on unlabelled + backward propagate to update parameters
            s_loss = criterion(s_outputs_u, pseudo_preds)
            s_loss.backward()
            s_optimizer.step()
            # s_scheduler.scale(s_loss).backward()
            # clip_gradient(s_optimizer, grad_clip) #Clip gradient to prevent exploding gradients
            # s_scaler.step(s_optimizer)
            # s_scaler.update()
            # s_scheduler.step()

            # Get dot product to feedback with teacher's unlabelled prediction
            s_outputs_l_new = s_model(images_l)
            s_loss_l_new = criterion(s_outputs_l_new, targets)
            dot_product = s_loss_l_old.detach() - s_loss_l_new.detach()

            # Calculate total teacher's loss
            t_loss_mpl = criterion(t_outputs_u.detach(), pseudo_preds.detach()) * dot_product
            t_loss = t_loss_l + t_loss_mpl

            # Update teacher parameters based on loss (feedback loop)
            t_loss.backward()
            t_optimizer.step()
            # t_scaler.scale(t_loss).backward()
            # clip_gradient(t_optimizer, grad_clip) #Clip gradient to prevent exploding gradients
            # t_scaler.step(t_optimizer)
            # t_scaler.update()
            # t_scheduler.step()

        s_scheduler.step()
        t_scheduler.step()

      # Validation phase.
      if phase == 'validation':
        s_model.eval()  # Set model to validation mode
        # t_model.eval()

        p_count = 0
        lowest_loss = 0.0
        running_loss = 0.0
        running_corrects = 0

        # Run prediction for student model using validation set.
        for inputs, labels in dataloaders[phase]:
          inputs = inputs.to(device)
          labels = labels.to(device)

          outputs = s_model(inputs)

          soft_preds = torch.softmax(outputs, dim=-1)  # Generate soft labels
          max_prob, hard_preds = torch.max(soft_preds, dim=-1)
          loss = criterion(soft_preds, labels)

          # Output training statistics.
          running_loss += loss.item() * inputs.size(0)
          running_corrects += torch.sum(hard_preds == labels.data)

        # Calculates epoch statistics.
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
          phase, epoch_loss, epoch_acc))

        # Add patience for early stopping
        if (lowest_loss <= epoch_loss):
          p_count += 1
          if p_count > patience:
            print(f'val loss did not decrease after {patience} epochs')
            break

        if lowest_loss > epoch_loss:
          p_count = 0
          lowest_loss = epoch_loss

        # Save best model weights if epoch gives best accuracy
        if epoch_acc >= best_acc:
          best_acc = epoch_acc
          best_epoch = epoch
          best_model_wts = copy.deepcopy(s_model.state_dict())

      epoch_end = time.time()
      epoch_time = epoch_end - epoch_start

      print()

  # Output runtime of model.
  time_elapsed = time.time() - since
  print('Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))
  with open(log_file, 'a') as log:
    log.write('Time {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    log.write('Best val Acc: {:4f} at epoch {}'.format(best_acc, best_epoch))

  # Save weights of the best epoch model into training directory.
  torch.save(best_model_wts, weight_file)

  return model
