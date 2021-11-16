#Function to create ensemble model, returns a dictionary of {model name: model weights}.
def createEnsemble(model_class, weights_path):
    #Pull files containing base model weights.
    file_paths = os.listdir(weights_path)

    #Retrieve weights of each base model.
    model_weights = list()
    for pth in file_paths:
        if 'pth' not in pth:
          continue
        path = f"{weights_path}/{pth}"
        model = model_class
        model.load_state_dict(torch.load(path, map_location = torch.device('cpu')), strict = False)
        model_weights.append(model)

    #Store base weights in dictionary.
    models = dict()
    models['model1'] = model_weights[0]
    models['model2'] = model_weights[1]
    models['model3'] = model_weights[2]
    models['model4'] = model_weights[3]
    models['model5'] = model_weights[4]

    #Delete intermediate variables to save memory
    del model_weights
    del model

    return models


#Function for ensemble prediction via majority voting classification.
def get_ensembleVoting(ensemble_model, test_input):
  #Initialise base models of ensemble model.
  model1 = ensemble_model['model1']
  model2 = ensemble_model['model2']
  model3 = ensemble_model['model3']
  model4 = ensemble_model['model4']
  model5 = ensemble_model['model5']
  
  #Send models to CPU/GPU 
  model1.to(device)
  model2.to(device)
  model3.to(device)
  model4.to(device)
  model5.to(device)

  #Set model to evaluation mode
  model1.eval()
  model2.eval()
  model3.eval()
  model4.eval()
  model5.eval()
  
  #Obtain base model predictions.
  preds = list()
  model1_output = model1(test_input)
  _, model1_pred = torch.max(model1_output.data, 1)
  preds.append(model1_pred)

  model2_output = model2(test_input)
  _, model2_pred = torch.max(model2_output.data, 1)
  preds.append(model2_pred)

  model3_output = model3(test_input)
  _, model3_pred = torch.max(model3_output.data, 1)
  preds.append(model3_pred)

  model4_output = model4(test_input)
  _, model4_pred = torch.max(model4_output.data, 1)
  preds.append(model4_pred)

  model5_output = model5(test_input)
  _, model5_pred = torch.max(model5_output.data, 1)
  preds.append(model5_pred)

  #Output final ensemble prediction.
  votes = collections.Counter(preds) #Counts frequency of votes
  common_vote = votes.most_common(1) #Retrive most common vote
  final_pred = common_vote[0][0]

  return final_pred
