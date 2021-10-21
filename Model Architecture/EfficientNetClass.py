class BaseEfficientNet(nn.Module):
  def __init__(self):
    super(BaseEfficientNet,self).__init__()
    #Import pre-trained b0 efficientnet - loads pre-trained weights for base model
    self.model = EfficientNet.from_pretrained('efficientnet-b0')

    self.avgpool = nn.AdaptiveAvgPool2d(1)
    self.dropout = nn.Dropout(0.5)
    self.fc = nn.Linear(1280, 2) 

  def forward(self, x):
    x = self.model.extract_features(x)
    # print(f"No of features: {x.shape}")
    x = self.avgpool(x)
    x = x.flatten(start_dim=1)
    x = self.dropout(x) #Randomly remove nodes to prevent overfitting
    x = self.fc(x)

    return x
