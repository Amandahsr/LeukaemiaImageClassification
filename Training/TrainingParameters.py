#Training parameters.
criterion = nn.CrossEntropyLoss() #Loss function.
seed = torch.manual_seed(42)
num_fold = 5 # to change
batch_size = 10
num_workers = 4
aug_rounds = 5
CV_states = 42
epochs = 100 # to change

#Create directory for training results
!mkdir /content/training_results/weights
!mkdir /content/training_results/stats
