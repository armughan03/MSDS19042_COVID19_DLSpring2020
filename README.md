# MSDS19042_COVID19_DLSpring2020
## COVID-19 Classification
In this, I have tried to experiment on few thousand X-Rays of Normal and COVID-19 patients. RESNET18, VGG16 are the two pretrained models that I have chosen for this and done two different experiments.
First was to simply fine tune the output layer according to the dataset while keeping the all other layers in  both models freezing and other was to unfreeze few CNN layers and then all the network.
In this notebook I have to train COVID19 data on pretrained RESNET18, VGG16 by doing Transfer learning and Fine tuning them. 

## Dataset:
New studies [1] have revealed that the damage done to lungs by infections belonging to the family of coronaviruses (COVID-19, SARS, ARDS etc.) can be observed in X-Ray and CT scan images. With a worldwide shortage of test kits, it is possible that careful analysis of X-Ray and CT scan images may be used in diagnosis of COVID-19 and for the regular assessment while they are recovering. In this assignment, we will use an open source dataset of X-Ray images and train a Convolutional Neural Network to try and detect instances of infections containing COVID-19 and Pneumonia.
### Dataset Details: 
This dataset contains X-Ray images from 2 classes:
 Copy to clipboard  Preview
| Class    	| # of images in training set 	| # of images in validation set 	| # of images in test set 	|
|----------	|-----------------------------	|-------------------------------	|-------------------------	|
| Infected 	| 4919                        	| 615                           	| 615                     	|
| Normal   	| 7081                        	| 885                           	| 885                     	|

![Image of Sample Dataset](Sample.png)
Dataset is available on this following link: https://drive.google.com/open?id=1-HQQciKYfwAO3oH7ci6zhg45DduvkpnK


## Trained Models:
All the trained models can be found at: https://drive.google.com/open?id=1Vh9kLEUMKrnnYqpGTaPxx8gA7j-Pyx90

### Analysis on each task and comparison of experiments to each other:
In task 1, when we only trained only last layer, it caused a lot of over fitting in both models and the accuracy was too high but again data for COVID19 is too small and too similar to pneumonia. While when I just unfreeze few CNN layers in task 2 models learn few more, but the overfitting was still the problem as we can see in the notebook and images above. Similarly even on fully unfreezed models was still causing the problems of overfitting but far less because of the data as it is too young and immature.

