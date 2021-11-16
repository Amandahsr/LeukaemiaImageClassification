#Testing function for base and ensemble models.
def test_model(model, stats_path, ensemble=False, fold = ''):
  #Message to indicate base or ensemble model evaluation.
  if ensemble == False:
    print("Evaluating base models...")
  else:
    print("Evaluating ensemble model...")
  
  #Create empty confusion matrix.
  confusion_matrix = np.zeros((2,2), dtype=int)
  
  with torch.no_grad():
    #Iterate over testing images.
    for inputs, labels in test_dataloader:
      inputs = inputs.to(device)
      labels = labels.to(device)
    
      if ensemble == False:
          output = model(inputs) #Base model prediction.
          _, prediction = torch.max(output.data, 1) #Hard labels.

      if ensemble == True:
          prediction = get_ensembleVoting(model, inputs) #Ensemble voting.

      #Add predictions to confusion matrix.
      for j in range(inputs.size()[0]): 
          if prediction[j]==1 and labels[j]==1:
              term='TP'
              confusion_matrix[0][0]+=1

          elif prediction[j]==1 and labels[j]==0:
              term='FP'
              confusion_matrix[1][0]+=1

          elif prediction[j]==0 and labels[j]==1:
              term='FN'
              confusion_matrix[0][1]+=1
        
          elif prediction[j]==0 and labels[j]==0:
              term='TN'
              confusion_matrix[1][1]+=1

      #Obtain results from confusion matrix.
      TP = confusion_matrix[0][0]
      FP = confusion_matrix[1][0]
      FN = confusion_matrix[0][1]
      TN = confusion_matrix[1][1]

      #Test statistics.
      accuracy = 100*(TP+TN)/ (TP+FP+TN+FN)
      sensitivity = (100*TP)/(TP+FP)
      specificity = (100*TN)/(TN+FN)
      PPV = (100*TP)/(TP+FN)
      NPV = (100*TN)/(TN+FP)
      F1 = 2*(PPV*sensitivity)/(PPV+sensitivity) 

    #Print test statistics.
    print('-----------------------')
    print('PREDICTION STATISTICS')
    print('-----------------------')
    print('True Positives: ' + str(TP))
    print('False Positives: ' + str(FP))
    print('False Negatives: ' + str(FN))
    print('True Negatives: ' + str(TN))

    print('-----------------------')
    print('EVALUATION STATISTICS')
    print('-----------------------')
    print('Accuracy: %f %%' % (accuracy))
    print('Sensitivity: %f %%' % (sensitivity))
    print('Specificity: %f %%' % (specificity))
    print('PPV: %f %%' % (PPV))
    print('NPV: %f %%' % (NPV))
    print('F1 Score: %f %%' % (F1))

    #Save test statistics as log files.
    stat_file = f"{stats_path}_stats.log"
    if ensemble:
      with open(stat_file, 'a') as log:
        log.write('ENSEMBLE MODEL:' + '\n')
    else:
      with open(stat_file, 'a') as log:
        log.write(fold + '\n')
        
    with open(stat_file, 'a') as log:
      log.write(
      'True Positives: ' + str(TP) + '\n'
      'False Positives: ' + str(FP) + '\n'
      'False Negatives: ' + str(FN) + '\n'
      'True Negatives: ' + str(TN) + '\n' + '\n'
      'Accuracy: %f %%' % (accuracy) +'\n'
      'Sensitivity: %f %%' % (sensitivity) + '\n'
      'Specificity: %f %%' % (specificity) + '\n'
      'PPV: %f %%' % (PPV) + '\n'
      'NPV: %f %%' % (NPV) + '\n'
      'F1 Score: %f %%' % (F1) + '\n')
