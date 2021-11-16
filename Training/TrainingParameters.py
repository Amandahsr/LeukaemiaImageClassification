#Training parameters.
learning_rate = 0.001
momentum = 0.9 # SGD momentum
step_size = 7 # scheduler step size
gamma = 0.1 # learning rate decay
seed = torch.manual_seed(42)
batch_size = 5
num_workers = 5
CV_states = 42
num_fold = 2
epochs = 1

#EfficientNet and Noisy Student model parameters.
base_model = BaseEfficientNet()
criterion = nn.CrossEntropyLoss() #Loss function.
optimizer = optim.SGD(base_model.parameters(), 
                      lr=learning_rate, 
                      momentum=momentum) #Stochastic Gradient descent.
scheduler = StepLR(optimizer, 
                   step_size=step_size, 
                   gamma=gamma) #Decay learning rate.
patience = 50 #Early stopping.

#Specify GPU/CPU usage.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model.to(device)
print(f"Device in use: {device}")
