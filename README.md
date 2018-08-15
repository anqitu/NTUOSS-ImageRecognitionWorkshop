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

Ensure "Hardware accelerator" is set to GPU (the default is CPU). Afterward, ensure that you are connected to the runtime (there is a green check next to "connected" in the menu ribbon).
<p align="center"> 
<img src="https://github.com/anqitu/NTUOSS-ImageRecognitionWorkshop/blob/master/screenshots/task_1_1_c.png" width="600">
</p>

To check whether you have a visible GPU (i.e. you are currently connected to a GPU instance), run the following code.
```python
# Task 1.1: Check if you are currently using the GPU in Colab
import tensorflow as tf
tf.test.gpu_device_name()
```

If you are connected, here is the response:
```
'/device:GPU:0'
```
And there you go. This allows you to access a free GPU for up to 12 hours at a time.

Alternatively, supply and demand issues may lead to this:
<p align="center"> 
<img src="https://github.com/anqitu/NTUOSS-ImageRecognitionWorkshop/blob/master/screenshots/task_1_1_d.png" width="600">
</p>
You will need to try again later to see whether there is any available free GPU. Don't worry even if you do not have the access to an available GPU now, as I will explain later.


#### 1.2 Mount Google Drive
<!-- TODO: screenshot for token -->

To import tha data into the VM, we will mount the google drive on the machine using `google-drive-ocamlfuse`.
(Reference: https://gist.github.com/Joshua1989/dc7e60aa487430ea704a8cb3f2c5d6a6)

```python
# Task: 1.2.1 Installing google-drive-ocamlfuse
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
```

Then, authenticate and get credentials for your googlr drive.
```python
# Task: 1.2.2 Authenticate and get credentials
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
```

Next, set up google-drive-ocamlfuse.
```python
# Task: 1.2.3 Setting up google-drive-ocamlfuse
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
```
At this step you will be asked two times to click on a link to allow access to your drive, at each step a token will be generated. Copy paste the token to the notebook.

And then you can finally mount your google drive.
```python
# TASK 1.2.4: Mount Google Drive in local Colab VM
!mkdir -p drive
!google-drive-ocamlfuse drive
!ls
```

Now, check working directory
```python
# TASK 1.2.5: Mount Google Drive in local Colab VM
!ls drive
```

```python
!ls drive/NTUOSS-ImageRecognitionWorkshop
```

```python
!ls drive/NTUOSS-ImageRecognitionWorkshop/data
```


#### 1.3 Project Setting
Finally, set up the path to the dataset.
```python
# TASK 1.3 : Set up the path to the dataset
import os
project_path = './drive/NTUOSS-ImageRecognitionWorkshop'
data_path = os.path.join(project_path, 'data')
image_path_train = os.path.join(data_path, 'train')
image_path_val = os.path.join(data_path, 'validation')
```


## Task 2 - Preprocess Images

#### 2.1 Configure Image Augmentation
This is the augmentation configuration we will use for training.

```python
# TASK 2.1: Create ImageDataGenerator
# This is the augmentation configuration we will use for training.

from keras.preprocessing.image import ImageDataGenerator

train_datagen =  ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
val_datagen = ImageDataGenerator(
    rescale=1. / 255)
```

#### 2.2 Generate Image Data from Directory
<!-- TODO: batch_size -->
<!-- TODO: class_mode -->

```python
# TASK 2.2 : Generate Image Data from Directory
# use .flow_from_directory() to generate batches of image data (and their labels) directly from our jpgs in their respective folders.

train_generator = train_datagen.flow_from_directory(
    image_path_train,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary')
val_generator = val_datagen.flow_from_directory(
    image_path_val,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary')

print('Class Indices : {}'.format(train_generator.class_indices))
```


## Task 3 - Build Basic Model

#### 3.1 Import the Keras libraries and packages
We’ve imported Conv2D from keras.layers, this is to perform the convolution operation i.e the first step of a CNN, on the training images. Since we are working on images here, which a basically 2 Dimensional arrays, we’re using Convolution 2-D,

We’ve imported MaxPooling2D from keras.layers, which is used for pooling operation, that is the step — 2 in the process of building a cnn. For building this particular neural network, we are using a Maxpooling function, there exist different types of pooling operations like Min Pooling, Mean Pooling, etc. Here in MaxPooling we need the maximum value pixel from the respective region of interest.

We’ve imported Flatten from keras.layers, which is used for Flattening. Flattening is the process of converting all the resultant 2 dimensional arrays into a single long continuous linear vector.

We’ve imported Dense from keras.layers, which is used to perform the full connection of the neural network, which is the step 4 in the process of building a CNN.

Let's build our first model, a simple stack of 3 convolution layers with a ReLU activation and followed by max-pooling layers
#### 3.2 Initialise Neural Network Model

#### 3.3 First Stack - Input Layer


<!-- TODO: Cnov/Relu/Pooling -->

#### 3.4 Second Stack

#### 3.5 Third Stack

#### 3.6 Output
<!-- TODO: Flatten/Dense/Putput -->

#### 3.7 Compile
<!-- TODO: loss/optimizer/metrics -->

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 148, 148, 32)      896       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 74, 74, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 72, 72, 64)        18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 36, 36, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 34, 34, 64)        36928     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 17, 17, 64)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 18496)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               4735232   
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 257       
=================================================================
Total params: 4,791,809
Trainable params: 4,791,809
Non-trainable params: 0
_________________________________________________________________
```

## Task 4 - Train Model
<!-- TODO: fit_generator -->

```
Epoch 1/25
  4/191 [..............................] - ETA: 1:25:10 - loss: 3.6668 - acc: 0.5078/usr/local/lib/python3.6/dist-packages/PIL/Image.py:872: UserWarning: Palette images with Transparency   expressed in bytes should be converted to RGBA images
  'to RGBA images')
191/191 [==============================] - 6835s 36s/step - loss: 0.7162 - acc: 0.6729 - val_loss: 0.5217 - val_acc: 0.7340
Epoch 2/25
191/191 [==============================] - 4319s 23s/step - loss: 0.5121 - acc: 0.7486 - val_loss: 0.3986 - val_acc: 0.8160
Epoch 3/25
191/191 [==============================] - 4332s 23s/step - loss: 0.4849 - acc: 0.7591 - val_loss: 0.4047 - val_acc: 0.8137
Epoch 4/25
191/191 [==============================] - 4419s 23s/step - loss: 0.4796 - acc: 0.7625 - val_loss: 0.4175 - val_acc: 0.8056
It takes 332.20 min to train the model
```

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
