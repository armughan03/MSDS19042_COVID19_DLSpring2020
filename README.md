# MSDS19042_COVID19_DLSpring2020
## COVID-19 Classification
In this, I have tried to experiment on few thousand X-Rays of Normal and COVID-19 patients. RESNET18, VGG16 are the two pretrained models that I have chosen for this and done two different experiments.
First was to simply fine tune the output layer according to the dataset while keeping the all other layers in  both models freezing and other was to unfreeze few CNN layers and then all the network.
In this assignment we have to train COVID19 data on pretrained RESNET18, VGG16. 

### Analysis on each task and comparison of experiments to each other:
In task 1, when we only trained only last layer, it caused a lot of over fitting in both models and the accuracy was too high but again data for COVID19 is too small and too similar to pneumonia. While when I just unfreeze few CNN layers in task 2 models learn few more, but the overfitting was still the problem as we can see in the notebook and images above. Similarly even on fully unfreezed models was still causing the problems of overfitting but far less because of the data as it is too young and immature.

