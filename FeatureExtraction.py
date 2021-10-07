#This script performs feature extraction on leukemia dataset using pre-trained EfficientNet-B0 model.

#Import pre-trained b0 efficientnet - loads pre-trained weights for base model
base_model = EfficientNet.from_pretrained('efficientnet-b0')

#Stop backpropagation by stopping calculation of gradients. Freezes all layers except last layer.
for param in base_model.parameters():
  param.requires_grad = False;

#Specify number of input and output. 
number_of_features = base_model.extract_features(image)
base_model.fc = nn.Linear(number_of_features, 2)

#Feed model to CPU.
base_model = base_model.to(device)

#Start optimising parameters of last layer of model using stochastic gradient descent.
optimizer = optim.SGD(base_model.parameters(), 
                           lr=learning_rate, momentum=momentum) #Should test different momentum values, incl 0

#Decay the learning rate by a factor of 0.1 for every 7 epochs
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

#Run training of model to obtain output layer.
trained_model = train_model(base_model, criterion, optimizer,
                         lr_scheduler, num_epochs=epochs)
