# Take a sample image
image = plt.imread("/content/ZB4171_LeukemiaImageClassification-Ongoing-/Data_main/images/UID_13_10_1_all.png")
image = np.moveaxis(image, -1, 0)
image = torch.from_numpy(image).type(torch.FloatTensor)
image = image.unsqueeze(0)


# #Specify number output layer.
base_model = BaseEfficientNet()
output = base_model(image)
print(f"Output shape: {output.shape}")
print(f"Output: {output}")

#Feed model to CPU.
base_model = base_model.to(device)

#Start optimising parameters of last layer of model using stochastic gradient descent.
optimizer = optim.SGD(base_model.parameters(), 
                           lr=learning_rate, momentum=momentum)

#Decay the learning rate by a factor of 0.1 for every 7 epochs.
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
