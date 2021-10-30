#Custom class for Ensembl model using models generated from all training folds.
class EnsblEfficientNet(nn.Module):
  def __init__(self, models, nb_classes=2):
    super(EnsblEfficientNet, self).__init__()
    self.modelA = models[0]
    self.modelB = models[1]
    self.modelC = models[2]
    self.modelD = models[3]
    self.modelE = models[4]

    # Remove last linear layer
    self.modelA.fc = nn.Identity()
    self.modelB.fc = nn.Identity()
    self.modelC.fc = nn.Identity()
    self.modelD.fc = nn.Identity()
    self.modelE.fc = nn.Identity()
      
    # Create new classifier
    self.classifier = nn.Linear(1280*5, nb_classes)
      
  def forward(self, x):
    #clone to make sure x is not changed by inplace methods
    x1 = self.modelA(x.clone()) 
    x1 = x1.view(x1.size(0), -1)
    x2 = self.modelB(x.clone())
    x2 = x2.view(x2.size(0), -1)
    x3 = self.modelC(x.clone())
    x3 = x3.view(x3.size(0), -1)
    x4 = self.modelD(x.clone())
    x4 = x4.view(x4.size(0), -1)
    x5 = self.modelE(x.clone())
    x5 = x5.view(x5.size(0), -1)

    #Concatenate models to form ensembl model.
    x = torch.cat((x1, x2, x3, x4, x5), dim=1)
  
  #Obtain prediction.
  x = self.classifier(F.relu(x))
  return x
