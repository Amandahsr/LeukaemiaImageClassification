#Import pre-trained b0 efficientnet - loads pre-trained weights for base model
base_model = EfficientNet.from_pretrained('efficientnet-b0')

#Specify number of input and output. 
number_of_features = base_model.extract_features(image)
base_model.fc = nn.Linear(number_of_features, output_layer)

#Feed model to CPU.
base_model = base_model.to(device)

#Start optimising parameters of last layer of model using stochastic gradient descent.
optimizer = optim.SGD(base_model.parameters(), 
                           lr=learning_rate, momentum=momentum)

#Decay the learning rate by a factor of 0.1 for every 7 epochs.
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
