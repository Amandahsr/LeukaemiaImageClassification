#Training function for EfficientNet model.
def efficientNet_training(model, criterion, optimizer, scheduler, num_epochs=epochs, fold=0): 
    # specify output files
    log_file = f"{baseEN_stats}/fold{fold+1}_log.log"
    weight_file = f"{baseEN_weights}/fold{fold+1}_weights.pth"

    since = time.time()

    #Initialise model.
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    #Epoch loop.
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_no = epoch+1
        print('Epoch {}/{}'.format(epoch_no, num_epochs))
        print('-' * 10)
        with open(log_file, 'a') as log:
          log.write('Epoch {}/{}, '.format(epoch_no, num_epochs))
        
        for phase in ['training', 'validation']:
            if phase == 'training':
                model.train() #Training mode.
                
            else:
                model.eval() #Validation mode.

            p_count = 0
            lowest_loss = 0.0
            running_loss = 0.0
            running_corrects = 0

            # Iterate over images.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                #Set parameter gradients to 0.
                optimizer.zero_grad()

                #Training phase.
                #Iterate forward.
                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs) #Model prediction.
                    soft_preds = torch.softmax(outputs, dim = -1) #Soft labels
                    max_prob, hard_preds = torch.max(soft_preds, dim = -1) #Hard labels
                    loss = criterion(soft_preds, labels) #Calculate loss.

                    #Backward propagate. 
                    if phase == 'training':
                        loss.backward()
                        optimizer.step()

                #Training statistics.
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(hard_preds == labels.data)
            
            if phase == 'training':
                scheduler.step()

            #Epoch statistics.
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            #Output training statistics.
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            with open(log_file, 'a') as log:
              log.write('{} Loss: {:.4f} Acc: {:.4f}, '.format(
                phase, epoch_loss, epoch_acc))
              
            #Validation phase.
            if phase == 'validation':
              with open(log_file, 'a') as log:
                  log.write('\n')

              #Add patience for early stopping.
              if (lowest_loss <= epoch_loss):
                p_count += 1
                if p_count > patience:
                  print(f'val loss did not decrease after {patience} epochs')
                  break

              if lowest_loss > epoch_loss:
                p_count = 0
                lowest_loss = epoch_loss

              #Save model weights if epoch gives best accuracy.
              if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

        #Output epoch statistics.
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        print('Epoch Time {:.0f}m {:.0f}s\n'.format(
          epoch_time // 60, epoch_time % 60))
        with open(log_file, 'a') as log:
          log.write('Time {:.0f}m {:.0f}s\n'.format(
          epoch_time // 60, epoch_time % 60))
        print()
    
    #Output runtime.
    time_elapsed = time.time() - since
    print('Time {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    with open(log_file, 'a') as log:
      log.write('Time {:.0f}m {:.0f}s\n'.format(
          time_elapsed // 60, time_elapsed % 60))
      log.write('Best val Acc: {:4f}'.format(best_acc))

    #Save weights of the best epoch model as file.
    torch.save(best_model_wts, weight_file)

    return model
