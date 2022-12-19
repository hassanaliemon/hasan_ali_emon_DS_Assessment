# Project Description
### N.B: Please download data and saved_models from [here](https://drive.google.com/file/d/1oP48LBBj6tq2t-DBWna4HPLFSjhsQIOt/view?usp=share_link) and replace with empty folders.
## Folder Description

1. data folder contains all data for train or test
2. logs folder contains project metrics ploted in png file
3. saved_models folder contains saved models that trained on given data
4. tensorboard folder contains tensorboard logs that can be monitored or visualized 


## Task 1: Preprocess the data
```
dataloader.py file preprocess the given data
```
## Task 2: Apply Augmentation
```
train_classifier.py file, line no 21 has augmentation implementation
```
## Build Classifier
```
get_model() function of train_classifier.py builds a classifier model with 4 classes
```
## Plot matrices
``` 
plot_loss_accuracy() function of train_classifier.py plots both loss and accuracy of model
```
## Model not Overfitted Proof
```
Model overfits when the model performes well in train data but very poor performance on test/val data. But if we look at the plotted graphs of accuracy or loss we can see that both our train and validation accuracy lies very close to each other as well as train and validation loss. This proves that our models did not overfitted. If it were then those graphs of train and validation accuracy lied at a distance from each other.
```
## Ensemble
```
Model ensemble implementation starts at line no 122 of train_classifier.py file
```

# Train the Classifier Model
Run the following command
```
python train_classifier.py
```
# Inference the Trained Model
Run the following command
```
python predict.py
```