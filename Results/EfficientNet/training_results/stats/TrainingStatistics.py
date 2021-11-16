# Script for plotting training curve per epoch
import os
import sys
import matplotlib.pyplot as plt

# input list of file for each fold
# log_files = sys.argv[1:]
# log_files = ['../fold1_v1_log.log'] # to change

log_dir = sys.argv[-1]
log_files = os.listdir(log_dir)

for file in log_files:
    epoch = []
    train_loss = []
    train_acc = []
    val_acc = []
    val_loss = []

    # read log file
    filepath = f"{log_dir}/{file}"
    with open(filepath, 'r') as log:

        for line in log.readlines()[:100]: #only plot first 100 epochs
            # get general info
            info = line.split(',')
            # get epoch info
            epoch_n = (info[0].split(' ')[-1]).split('/')[0]
            epoch.append(int(epoch_n))
            # get training info
            training = info[1]
            training_l = training.split(':')[1].split(' ')[1]
            training_a = training.split(':')[2].split(' ')[-1]
            train_loss.append(float(training_l))
            train_acc.append(float(training_a))
            # get validation info
            training = info[2]
            valid_l = training.split(':')[1].split(' ')[1]
            valid_a = training.split(':')[2].split(' ')[-1]
            val_loss.append(float(valid_l))
            val_acc.append(float(valid_a))

    # plotting
    fig = plt.figure(figsize=(5, 6))
    # loss subplot
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(epoch, val_loss, color='blue', label='val loss')
    ax.plot(epoch, train_loss, color='orange', label='train loss')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc='upper right',
              fontsize='small',
              bbox_to_anchor=(1.28, 1))
    # accuracy subplot
    ax = fig.add_subplot(2, 1, 2)
    ax.plot(epoch, val_acc, color='blue', label='val accuracy')
    ax.plot(epoch, train_acc, color='orange', label='train accuracy')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='upper right',
              fontsize='small',
              bbox_to_anchor=(1.35, 1))
    # plt.show()
    filename = file.split('/')[-1].split('.')[0]
    fig.savefig(log_dir + '/' + filename + '.png' , bbox_inches="tight")
