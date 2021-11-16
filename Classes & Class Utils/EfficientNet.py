class BaseEfficientNet(nn.Module):
  def __init__(self):
    super(BaseEfficientNet,self).__init__()
    self.model = EfficientNet.from_pretrained('efficientnet-b0') #Import pre-trained EfficientNet-B0.

    self.avgpool = nn.AdaptiveAvgPool2d(1)
    self.dropout = nn.Dropout(0.5)
    self.fc = nn.Linear(1280, 2) 

  def forward(self, x):
    x = x.float()
    x = self.model.extract_features(x)
    x = self.avgpool(x)
    x = x.flatten(start_dim=1)
    x = self.dropout(x) #Randomly remove nodes to prevent overfitting
    x = self.fc(x)

    return x
