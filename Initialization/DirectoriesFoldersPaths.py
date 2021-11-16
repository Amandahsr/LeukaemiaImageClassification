#Mount google drive to store datasets and results in.
drive.mount('/content/drive')

#Directory to store base EfficientNet model information
!mkdir /content/baseEfficientNet
!mkdir /content/baseEfficientNet/training_results
!mkdir /content/baseEfficientNet/training_results/weights #Weights
!mkdir /content/baseEfficientNet/training_results/stats #Training stats
!mkdir /content/baseEfficientNet/testing_results/baseStats #Testing stats for base models
!mkdir /content/baseEfficientNet/testing_results/ensembleStats #Testing stats for ensemble model

#Directory to store Noisy Student model information
!mkdir /content/noisyStudent
!mkdir /content/noisyStudent/training_results
!mkdir /content/noisyStudent/training_results/weights #Weights
!mkdir /content/noisyStudent/training_results/stats #Training stats
!mkdir /content/noisyStudent/testing_results/baseStats #Testing stats for base models
!mkdir /content/noisyStudent/testing_results/ensembleStats #Testing stats for ensemble model

#File paths for training.
train_labels = "LeukemiaData/train_labels.csv"
train_images = "LeukemiaData/train_images"
test_labels = "LeukemiaData/testing/test_labels.csv"
test_images = "LeukemiaData/testing/test_images"

#EfficientNet model
baseEN_weights = "baseEfficientNet/training_results/weights"
baseEN_stats = "baseEfficientNet/training_results/stats"
baseEN_test = "/content/baseEfficientNet/testing_results/baseStats"
ensembleEN_test = "/content/baseEfficientNet/testing_results/ensembleStats"

#Noisy Student model
baseNS_weights = "/content/noisyStudent/training_results/weights"
baseNS_stats = "/content/noisyStudent/training_results/stats"
baseNS_test = "/content/noisyStudent/testing_results/baseStats"
ensembleNS_test = "/content/noisyStudent/testing_results/ensembleStats"
