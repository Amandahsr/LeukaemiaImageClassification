# Leukaemia Image Classification

The Leukaemia Image Classification project is a joint collaboration between [Amanda Ho](https://github.com/Amandahsr) and [Tiana Chen](https://github.com/Tianananana). 

This project aims to build an image classification model for leukemia cell detection in microscopic images. This project is built upon the [Leukemia Classification Challenge](https://www.kaggle.com/andrewmvd/leukemia-classification) available in Kaggle. All code is implemented and tested on Google Colab using Pytorch packages.

Methodology:
1. Training and validation datasets are merged and split into training and testing sets via patient ID.
2. EfficientNet models are trained using K-fold cross validation and nested epochs.
3. Ensemble EfficientNet model is implemented using the trained EfficientNet models as base models. Ensemble model uses majority voting as voting system.
4. As a further improvement, Noisy Student training is implemented in addition to EfficientNet model. Noisy student training is carried out within a K-fold cross validation and nested epoch loop.
5. Ensemble Noisy Student model is implemented using the trained Noisy Student models as base models. Ensemble model uses majority voting as voting system.
6. The 4 models implemented in the project (EfficientNet, Ensemble EfficientNet, Noisy Student, Ensemble Noisy Student) are tested and evaluated for performance.

![](https://github.com/Amandahsr/ZB4171_LeukemiaImageClassification-Ongoing-/blob/main/Project%20Overview.png) 
