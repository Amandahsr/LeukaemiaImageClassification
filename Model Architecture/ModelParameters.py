#Model parameters.
criterion = nn.CrossEntropyLoss() #Loss function.
learning_rate = 0.001
momentum = 0.9
output_layer = 2

#Training parameters.
seed = torch.manual_seed(42)
# seed = 42
num_fold = 5 # to change
batch_size = 10
num_workers = 4
aug_rounds = 5
CV_states = 42

#Use CUDA to run training, else use CPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
