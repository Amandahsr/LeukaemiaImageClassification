#Ensembl code for MPL model.
#Pull all files containing weights from each training fold.
dir = "/content/MetaPseudoModel/training_results/weights"
file_paths = os.listdir(dir)

#Store models weights in dictionary.
metapseudo_models = {}
for file in file_paths:
  path = dir + file
  model = EfficientNet()
  model.load_state_dict(torch.load(path), strict = False)
  models[file.split('.')[0]] = model

#Create MPL ensembl model.
mpl_models = metapseudo_models.values()
mpl_ensembl_model = EnsblEfficientNet(mpl_models)
