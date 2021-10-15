# Leukaemia Image Classification

A collaborative effort between Amanda Ho and Tiana Chen to build an image classification model for leukaemia cell detection in microscopic images. This project is built upon the open challenge [Leukemia Classification](https://www.kaggle.com/andrewmvd/leukemia-classification) in Kaggle.

Project code is implemented on Google Colab using Pytorch libraries.

Methodology:
1. Import leukemia image data to Google Colab from Kaggle.
2. Perform data augmentation to generate more data for training, validation and testing.
3. Train using pre-trained EfficientNet-B0 model to feature extract from leukemia dataset. 
    - Stratified k-fold cross validation + epoch is implemented, using patient ID to split training and validation phases in each epoch.
    - Output layer is changed to binary classification.
5. Implement Iterative Teacher-Student model using trained EfficientNet model.
6. Carry out testing on model.
