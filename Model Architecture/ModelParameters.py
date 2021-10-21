#Model parameters.
learning_rate = 0.001
momentum = 0.9
output_layer = 2

#Use CUDA to run training, else use CPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
