#Training base EfficientNet model function. Runs a specified number of epochs.
def train_model(model, criterion, optimizer, scheduler, num_epochs=epochs):
    since = time.time()

    #Initialise model.
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['training', 'validation']:
            if phase == 'training':
                model.train()  # Set model to training mode
                
            else:
                model.eval()   # Set model to validation mode
                

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                #Set parameter gradients to 0.
                optimizer.zero_grad()

                #Iterate forward.
                #Track history only if in training phase.
                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs)
                    soft_preds = torch.softmax(outputs, dim = -1) #Generate soft labels
                    max_prob, hard_preds = torch.max(soft_preds, dim = -1)
                    loss = criterion(soft_preds, labels)

                    #Backward propagate. 
                    #Update parameters only if in training phase.
                    if phase == 'training':
                        loss.backward()
                        optimizer.step()

                #Output training statistics.
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(hard_preds == labels.data)
            
            if phase == 'training':
                scheduler.step()

            #Calculates epoch statistics.
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            #Initialise model if in validation phase and epoch has best accuracy.
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
    
    #Output runtime of model.
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    #Load weights of model.
    model.load_state_dict(best_model_wts)

    return model
