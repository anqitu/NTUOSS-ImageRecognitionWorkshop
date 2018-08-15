# NTUOSS Image Recognition Workshop

*by [Tu Anqi](https://github.com/anqitu) for NTU Open Source Society*

This workshop assumes beginner-level knowledge of Python.

**Disclaimer:** *This document is only meant to serve as a reference for the attendees of the workshop. It does not cover all the concepts or implementation details discussed during the actual workshop.*
___

### Workshop Details
**When**: Friday, 31 Aug 2018. 6:30 PM - 8:30 PM.</br>
**Where**: @TODO </br>
**Who**: NTU Open Source Society

### Questions
Please raise your hand any time during the workshop or email your questions to [me](mailto:anqitu@outlook.com) later.

### Errors
For errors, typos or suggestions, please do not hesitate to [post an issue](https://github.com/anqitu/NTUOSS-ImageRecognitionWorkshop/issues/new). Pull requests are very welcome! Thanks!
___

## Task 0 - Getting Started

#### 0.1 Introduction

<!-- TODO: write about cnn -->
<!-- TODO: write about keras -->
<!-- TODO: write about colab -->

For this tutorial, we'll be creating a Convolutional Neural Network(CNN) model with Keras on Colaboratory. The model will be able to classify the images of cat and dog.

1. What are [CNNs](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)?\
Convolutional Neural Networks (CNNs or ConvNets) are a category of [Neural Networks](https://ujjwalkarn.me/2016/08/09/quick-intro-neural-networks/) that have proven very effective in areas such as image recognition and classification. CNNs have been successful in identifying faces, objects and traffic signs apart from powering vision in robots and self driving cars.

2. What is [Keras](https://keras.io/)?\
Keras is an open source neural network library written in Python. It was developed with a focus on enabling fast experimentation.

3. What is [Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)?\
Colaboratory is a Google research project created to help disseminate machine learning education and research. It is a free Jupyter notebook environment that requires no setup and runs entirely in the cloud.

#### 0.2 Initial Setup

1.  Add this [folder](https://drive.google.com/open?id=1uZT-vRnWgxYp9wgzYw6tTPS_lW20T9e7) to your google drive

Inside the folder, you will find 3 folders:

```
/NTUOSS-ImageRecognitionWorkshop
  /data
  /complete
  /start
```

In the /data folder, there are train, test and validation image folders, with the data distribution shown as below.

```
/data
  /train:
    /cat: 3304
    /dog: 3601
  /validation
    /cat: 1400
    /dog: 1546
  /test
    /cat: 537
    /dog: 584
```

<!-- TODO: explain train vs validation vs test-->
The model is initially fit on a training dataset, which is a set of examples used to fit the parameters of the model (e.g. weights of connections between neurons in neural networks). The training dataset often consists of pairs of an input vector and the corresponding target. In our case, each image is an input vector, while the image's label (dog or cat) is a target.

Then, the fitted model is used to predict the responses for the observations in the validation dataset. The validation dataset provides an unbiased evaluation of the fitted model. It can be used for regularization by early stopping: stop training when the error on the validation dataset increases, as this is a sign of overfitting to the training dataset.

Finally, the test dataset is a dataset used to provide an unbiased evaluation of a final model fit on the training dataset.

The /complete folder contains complete codes for this project, including extracting image urls, downloading images, training model and predicting images. This workshop will focus on the model training and image recognition part. For those who are interested in how I crawl all the images from google, do take a look on the other two scripts.

```
/complete
  /model
  /0_Extract_Image_url.py
  /1_Image_Downloader.py
  /2_Train_Model.py
  /3_Predict.py
  /3_Recognize_Image.py
  /util.py
```

This /start folder contains the incomplete codes for the purpose of this workshop. Now, let's start by opening the Train_Model.py file with colaboratory.

```
/start
  /Train_Model.py
  /Predict.py
```


![task 0.2 screenshot a](screenshots/task_0_2.png?raw=true)

## Task 1 - Virtual Environment

#### 1.1 Change to a Free GPU Runtime
<!-- TODO: write about CPU vs GPU -->
<!-- TODO: screenshot of changing GPU -->
Apart from saving us trouble in setting up environments, Colab also provides free GPU that speeds up the training and prevents your own laptop from overheating.
Select "Runtime," "Change runtime type,".

<p align="center"> 
<img src="https://github.com/anqitu/NTUOSS-ImageRecognitionWorkshop/blob/master/screenshots/task_1_1_a.png" width="600">
</p>

On this pop-up, select GPU.
<p align="center"> 
<img src="https://github.com/anqitu/NTUOSS-ImageRecognitionWorkshop/blob/master/screenshots/task_1_1_b.png" width="600">
</p>

#### 1.2 Mount Google Drive
<!-- TODO: screenshot for token -->

![task 1.2 screenshot a](screenshots/task_1_2_a.png?raw=true)

![task 1.2 screenshot b](screenshots/task_1_2_b.png?raw=true)

#### 1.3 Project Setting

![task 1.3 screenshot a](screenshots/task_1_3_a.png?raw=true)

![task 1.3 screenshot b](screenshots/task_1_3_b.png?raw=true)

## Task 2 - Preprocess Images

#### 2.1 ImageDataGenerator

#### 2.2 Image Augmentation

#### 2.3 flow_from_directory
<!-- TODO: batch_size -->
<!-- TODO: class_mode -->



## Task 3 - Build Basic Model
<!-- TODO: VGG -->
# VGG-like convnet from Keras: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

# A simple cnn model, with 3 simple stacks of 2 convolution layers with a ReLU activation and followed by max-pooling layers.
# cnn: https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/

#### 3.1 Sequential

#### 3.2 First Stack - Input Layer
<!-- TODO: Cnov/Relu/Pooling -->

#### 3.3 Second Stack

#### 3.4 Third Stack

#### 3.5 Output
<!-- TODO: Flatten/Dense/Putput -->

#### 3.5 Compile
<!-- TODO: loss/optimizer/metrics -->


## Task 4 - Train Model
<!-- TODO: fit_generator -->

## Task 5 - Test Model


## Task 6 - Recognize Image

## Task 7 - Build Advanced Model
<!-- TODO: bottle neck -->

___

## Acknowledgements

Many thanks to [clarencecastillo](https://github.com/clarencecastillo) for carefully testing this walkthrough and to everybody else in [NTU Open Source Society](https://github.com/ntuoss) committee for making this happen! :kissing_heart::kissing_heart::kissing_heart:

## Resources
[Keras Docs]
[VGG16 Docs]
[Colab Docs]
