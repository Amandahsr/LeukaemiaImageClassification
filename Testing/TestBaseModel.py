#Testing function.
def test_model(dataloader, model, criterion):
  #Create empty confusion matrix.
  confusion_matrix = np.zeros((2,2),dtype=int)

  with torch.no_grad():
    #Iterate over testing data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        #Test model.
        output = model(inputs)
        _, prediction = torch.max(output.data, 1)

        #Obtain predictions and initialise in confusion matrix.
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

        TP = confusion_matrix[0][0]
        FP = confusion_matrix[0][1]
        FN = confusion_matrix[1][0]
        TN = confusion_matrix[1][1]

        #Calculate test statistics.
        accuracy = 100*(TP+TN)/ (TP+FP+TN+FN)
        sensitivity = (100*TP)/(TP+FP)
        specificity = (100*TN)/(TN+FN)
        PPV = (100*TP)/(TP+FN)
        NPV = (100*TN)/(TN+FP)
        F1 = 2*(PPV*sensitivity)/(PPV+sensitivity) 

    #Print test statistics.
    print('Accuracy: %d %%' % (accuracy))
    print('Sensitivity: %d %%' % (sensitivity))
    print('Specificity: %d %%' % (specificity))
    print('PPV: %d %%' % (PPV))
    print('NPV: %d %%' % (NPV))
    print('F1 Score: %d %%' % (F1))

#Initialise testing set files
test_path = "/content/ZB4171_LeukemiaImageClassification-Ongoing-/Data_main/held_out_test/test_labels.csv"
test_df = pd.read_csv(test_path)[["Patient_no", "Patient_ID", "labels"]]
test_set = CellsDataset(test_df, "/content/ZB4171_LeukemiaImageClassification-Ongoing-/Data_main/held_out_test/test_images")

#Dataloader for input into the model.
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

#Test model.
test_model(test_loader, base_model, criterion)
