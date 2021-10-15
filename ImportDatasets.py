##This script imports datasets from the Leukemia Classification challenge directly from Kaggle into Google Colab.

#Mount google drive to store datasets and results in.
drive.mount('/content/drive')

#Upload API token file downloaded from Kaggle website
from google.colab import files
files.upload()

#Make a new directory and copy API token file into directory
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/

#Change permission of file
! chmod 600 ~/.kaggle/kaggle.json
! kaggle datasets list

#Download datasets from Leukaemia challenge
! kaggle datasets download -d andrewmvd/leukemia-classification

#Make directory to unzip and store downloaded datasets
! mkdir datasets
! unzip leukemia-classification.zip -d datasets
