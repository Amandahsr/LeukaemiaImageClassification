#Convert testing set to custom class.
test_df = pd.read_csv(test_labels)[["Patient_no", "Patient_ID", "labels"]]
test_set = CellsDataset(test_df, test_images)

#Send testing set to dataloader.
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

#Evaluate base EfficientNet models.
EN_weights = os.listdir(baseEN_weights)
for weight in EN_weights:
  pth = weight_dir + '/'+ weight
  model = BaseEfficientNet()
  model.load_state_dict(torch.load(pth))
  model.to(device) #load model to gpu

  path = baseEN_stats + '/' + weight.split('.')[0]
  test_model(model, stats_path=path, ensemble=False)

#Evaluate base Noisy Student models.
NS_weights = os.listdir(baseNS_weights)
for weight in NS_weights:
  pth = weight_dir + '/'+ weight
  model = BaseEfficientNet()
  model.load_state_dict(torch.load(pth))
  model.to(device) #load model to gpu

  path = baseNS_stats + '/' + weight.split('.')[0]
  test_model(model, stats_path=path, ensemble=False)

#Evaluate ensemble models.
test_model(ensembleEN_model, stats_path=ensembleEN_test, ensemble=True)
test_model(ensembleNS_model, stats_path=ensembleNS_test, ensemble=True)
