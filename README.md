# NTUOSS Image Recognition Workshop

*by [Tu Anqi](https://github.com/anqitu) for NTU Open Source Society*

This workshop assumes basic knowledge of Python.

**Disclaimer:** *This document is only meant to serve as a reference for the attendees of the workshop. It does not cover all the concepts or implementation details discussed during the actual workshop.*
___

### Workshop Details
**When**: Friday, 31 Aug 2018. 6:30 PM - 8:30 PM.</br>
**Where**: LT1 </br>
**Who**: NTU Open Source Society

### Questions
Please raise your hand any time during the workshop or email your questions to [me](mailto:anqitu@outlook.com) later.

### Errors
For errors, typos or suggestions, please do not hesitate to [post an issue](https://github.com/anqitu/NTUOSS-ImageRecognitionWorkshop/issues/new). Pull requests are very welcome! Thanks!
___

## Task 0 - Getting Started

#### 0.1 Introduction

For this tutorial, we'll be creating a Convolutional Neural Network(CNN) model with Keras on Colaboratory. The model will be able to classify the images of cat and dog.

1. What is Image Recognition (or Image Classification)?\
When a computer sees an image (takes an image as input), it will see an array of pixel values. Depending on the resolution and size of the image, it will see a WIDTH x HEIGHT x 3 array of numbers (The WIDTH and HEIGHT refers to the size while the 3 refers to RGB values). For example, suppose we have a colorful image in JPG format with a size 480 x 480. The array seen by the computer will be 480 x 480 x 3. Each of these numbers is a value between 0 and 255 which describes the pixel intensity at that point. These numbers, while meaningless to us when we perform image classification, are the only inputs available to the computer.\
The idea of image classification is that, you give the computer this array of numbers, then it will output numbers that describe the probability of the image being a certain class (eg. 0.8 for cat, 0.2 for dog).</br>

<p align="center">
  <img src="https://github.com/anqitu/NTUOSS-ImageRecognitionWorkshop/blob/master/screenshots/task_0_1.png" width="500">
</p> 


2. What are [CNNs](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)?\
Convolutional Neural Networks (CNNs or ConvNets) are a category of [Neural Networks](https://ujjwalkarn.me/2016/08/09/quick-intro-neural-networks/) that are very effective in areas such as image recognition and classification. They have been successful in identifying faces, objects and traffic signs apart from powering vision in robots and self driving cars.

3. What is [Keras](https://keras.io/)?\
Keras is an open source neural network library written in Python. It was developed with a focus on enabling fast experimentation.

4. What is [Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)?\
Colaboratory is a Google research project created to help disseminate machine learning education and research. It is a free Jupyter notebook environment that requires no setup and runs entirely in a virtual machine (VM) hosted in the cloud.



#### 0.2 Initial Setup

Add this [folder](https://drive.google.com/open?id=1uZT-vRnWgxYp9wgzYw6tTPS_lW20T9e7) to your google drive

<p align="center">
  <img src="https://github.com/anqitu/NTUOSS-ImageRecognitionWorkshop/blob/master/screenshots/task_0_2_a.png" width="500">
</p>

Inside the folder, you will find one data folder and one start file:
```
/NTUOSS-ImageRecognitionWorkshop
  /data
  /start
```

In the /data folder, there are train, test and validation image folders, with the data distribution shown as below. To allow Keras to use its special API to handle the data downloads directly from the folder, the tructure of the project folder must be as following. There is a also a model folder containing the models I have trained before this workshop.

```
/data
  /train:
    /cat: 2000
    /dog: 2000
  /validation
    /cat: 1000
    /dog: 1000
  /test
    /cat: 100
    /dog: 100
  /model
    /cnn_model_basic.h5
    /cnn_model_advanced.h5
```
Here are the purposes of each type of data set:
- **Train**: The model is initially fit on a training dataset, which is a set of examples used to fit the parameters of the model (e.g. weights of connections between neurons in neural networks). The training dataset often consists of pairs of an input vector and the corresponding target. In our case, each image is an input vector, while the image's label (dog or cat) is a target.

- **Validation**: The validation dataset provides an unbiased evaluation of the fitted model. It can be used for regularization by early stopping: stop training when the error on the validation dataset increases, as this is a sign of overfitting to the training dataset.

- **Test**: The test dataset is a dataset used to provide an unbiased evaluation of our final model.

This 'start' file is Colab Notebooks which contains the incomplete codes for the purpose of this workshop.

Now, let's open the start file to officially start the coding part of today's workshop: Right click start file -> Select 'Open with' -> Select 'Colaboratory'.

<p align="center">
  <img src="https://github.com/anqitu/NTUOSS-ImageRecognitionWorkshop/blob/master/screenshots/task_0_2_b.png">
</p> 


## Task 1 - Virtual Environment

#### 1.1 Change to a Free GPU Runtime
<!-- TODO: write about CPU vs GPU -->
Apart from saving us trouble in setting up environments, Colab also provides free GPU that speeds up the training and prevents your own laptop from overheating.

First of all, we need to ensure our "Hardware accelerator" is set to GPU (the default is CPU): Select "Runtime," -> "Change runtime type,".

<p align="center">
  <img src="https://github.com/anqitu/NTUOSS-ImageRecognitionWorkshop/blob/master/screenshots/task_1_1_a.png" width="500">
</p> 


On this pop-up, select GPU.

<p align="center">
  <img src="https://github.com/anqitu/NTUOSS-ImageRecognitionWorkshop/blob/master/screenshots/task_1_1_b.png" width="500">
</p> 


 Afterward, ensure that you are connected to the runtime (there is a green check next to "connected" in the menu ribbon).

<p align="center">
  <img src="https://github.com/anqitu/NTUOSS-ImageRecognitionWorkshop/blob/master/screenshots/task_1_1_c.png" width="500">
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
  <img src="https://github.com/anqitu/NTUOSS-ImageRecognitionWorkshop/blob/master/screenshots/task_1_1_d.png" width="500">
</p> 

You will need to try again later to see whether there is any available free GPU. Don't worry even if you do not have the access to an available GPU now, as I will explain later.


#### 1.2 Mount Google Drive

To import tha data into the VM, we will mount the google drive on the machine using `google-drive-ocamlfuse`. ([Reference](https://gist.github.com/Joshua1989/dc7e60aa487430ea704a8cb3f2c5d6a6))

```python
# Task: 1.2.1 Install google-drive-ocamlfuse
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
```

Then, authenticate and get credentials for your google drive.
```python
# Task: 1.2.2 Authenticate and get credentials
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()

import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
```
You will be asked two times to authenticate the access to your drive. At each step a token will be generated:
- Click on the link to log into your google account.
- Allow access to your drive.
- Copy the token (The token looks like this - 4/PABmEY7BRPd3jPR9BI9I4R99gc9QITTYFUVDU76VR)
- Switch to the notebook to paste the token in the 'Enter verification code' bar.
- Press 'Enter'

And then you can mount your google drive in your current virtual machine.
```python
# TASK 1.2.3: Mount Google Drive in local Colab VM
!mkdir -p drive
!google-drive-ocamlfuse drive
!ls
```

You should see a /drive folder inside the current working directory.
```
adc.json  datalab  drive  sample_data
```

Running the code below should let you see the folders inside your google drive.
```python
!ls drive
```

Then, access our working directory /drive/NTUOSS-ImageRecognitionWorkshop by running the code below.
```python
!ls drive/NTUOSS-ImageRecognitionWorkshop
```

Lastly, check the /data directory.
```python
!ls drive/NTUOSS-ImageRecognitionWorkshop/data
```


## Task 2 - Preprocess Images

#### 2.1 Configure Image Augmentation
<!-- TODO: image augmentation -->
As we only have 2000 training images for each class, this is considered definitely quite few for the model to learn enough patterns and recognize images accurately. (For example, [VGG16](https://arxiv.org/abs/1409.1556) is a convolutional neural network model trained on 14 million images to recognize an image as one of 1000 categories with an accuracy of 92.5%.) One way to enlarge our existing dataset is some easy transformations. As previously mentioned, when a computer takes an image as an input, it will take in an array of pixel values. Imagine that the whole image is shifted left by 1 pixel. For us, this change is imperceptible. However, for a computer, this shift can be indeed very significant. Approaches that alter the training data in ways that change the array representation while keeping the label the same are known as **data augmentation**. It allows us to artificially expand our dataset. Some popular augmentations are horizontal flips, random crops, translations, rotations, and so on. By applying just a couple of these transformations to our training data, we can easily enlarge our data set.

Here we import the [```ImageDataGenerator```](https://keras.io/preprocessing/image/) from Keras libraries and add some data augmentation parameters for the image data generator.
- **rescale = 1. / 255**: Rescaling factor. The factor to multiply every pixel in the preprocessing image. As mentioned earlier, each digital colorful image contains three maps of pixels: Red, Green and Blue, and all the pixels are in the range 0~255. Since 255 is the maximin pixel value. Rescale 1./255 is to transform every pixel value from range [0,255] -> [0,1]. The benefit of such a rescaling is that it makes the model treat all images (regardless with high or low pixel range) in the same manner.
- **rotation_range = 30**: Int. Degree range for random rotations
- **width_shift_range = 0.2**: Float Fraction range of total width for shifting.
- **height_shift_range = 0.2**: Float. Fraction range of total heighth for shifting.
- **zoom_range = 0.2**: Float. Fraction range for random zoom.
- **horizontal_flip = True**: Boolean. Randomly flip inputs horizontally.

```python
# TASK 2.1 : Add augmentation configuration for the data generator of train data only
datagen_train =  ImageDataGenerator(
    rescale = 1. / 255,
    rotation_range = 30,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)
```

#### 2.2 Generate Image Data from Directory
Now, let's use [```.flow_from_directory```](https://keras.io/preprocessing/image/) to generate batches of image data (and their labels) directly from our jpgs in their respective folders. This is another great API provided by Keras which saves us efforts in converting images to pixel arrays before feeding our model.

Also it is time to put some additional parameters, like class_mode, target_size and batch_size.
- **directory**: Path to the target directory. It should contain one subdirectory per class
- **target_size = (150, 150)**: Size of image to be fed to our model. As all images from our data set has different resolution, we need to standardize their size before feeding our model. It must in a [tuple](https://www.tutorialspoint.com/python/python_tuples.htm). Here we set it as (150, 150).
- **class_mode = 'binary'**: Mode of class. Set as binary since the model is trained to classify whether the image is a cat a dog, which is a yes/no question. If the model is trained on classifying more than 2 categories, the class mode should be set to 'categorical'.
- **shuffle = True**: Whether to shuffle the data. Set True if you want to shuffle the order of the image that is being yielded, else set False.
- **batch_size = 50**: Number of images in one batch of data. Here we need to explain a bit on the concept of epoch and batch size:\
`one epoch` = one forward pass and one backward pass of all the training samples.\
`batch size` = the number of training samples in one forward/backward pass. Each batch size will be used to update the model parameters. Ideally, we would want to use all the training samples to calculate the gradients for every single update, but that is not efficient. Also, the higher the batch size, the more memory space we'll need. One strategy is to split the data into batches and fit them one by one to the model to update the parameters.\
`One pass` = one forward pass + one backward pass.\
For instance, here we have 4000 training samples and we set the batch_size as 50. The first 50 samples from the training dataset will be used to train the network. Then it takes second 50 samples to train network again. This procedure continues until all samples have been propagated through the networks. In total, there will be 4000/50 = 80 batches for each epoch. We select 50 here because it divides 2000.


```python
# TASK 2.2 : Generate Image Data from Directory and Set parameter
train_data = datagen_train.flow_from_directory(
    directory = './drive/NTUOSS-ImageRecognitionWorkshop/data/train',
    target_size = (150, 150),
    class_mode = 'binary',
    shuffle = True,
    batch_size = 50)
validation_data = datagen_val.flow_from_directory(
    directory = './drive/NTUOSS-ImageRecognitionWorkshop/data/validation',
    target_size = (150, 150),
    class_mode = 'binary',
    shuffle = True,
    batch_size = 50)
```

As we have 2000 images of each class for train set and 1000 images of each class for validation set, you should see this response:
```
Found 4000 images belonging to 2 classes.
Found 2000 images belonging to 2 classes.
```

Then, running the code below will let you see the class indices of the data generator.
```python
# TASK 2.2.2 : Check class indices
print('Class Indices : {}'.format(train_data.class_indices))
```

You will see the response as below.
```
Class Indices : {'cat': 0, 'dog': 1}
```
This class indices imply that, ideally, the model will predict the image as 0 if it is cat, and predict the image as 1 if it is a dog. However, the model will only be able to predict the probability of whether the image is a cat or a dog. Thus, the closer the probability is to 0, the higher the confidence that the image is a cat. The closer the probability is to 1, the higher the confidence that the image is a dog. For example, a probability of 0.8 indicates a dog, while a probability of 0.2 indicates a cat.


## Task 3 - Build Basic Model

#### 3.1 Set up backend and Import libraries
<!-- TODO: backend -->

Keras is a model-level library that provides high-level building blocks for developing deep learning models. Instead of handling itself low-level operations such as tensor products, convolutions and so on, it relies on other specialized libraries to do so. Several different backend engines can be plugged seamlessly into Keras.

For this workshop, we will use [TensorFlow](https://www.tensorflow.org/) as the backend, which is an open-source framework developed by Google. Here, we set the value of the image dimension ordering to 'tf' (stands for tensorflow) which uses 'channels_last'. For 2D data (e.g. image), "channels_last" assumes (rows, cols, channels) while "channels_first" assumes  (channels, rows, cols). Thus, we will also need to set the image data format as 'chennels_last' so that it complies with our image data format.

```python
# TASK 3.1.1 Configure backend
from keras import backend as K
K.set_image_dim_ordering('tf') #channel last
K.set_image_data_format('channels_last')
```


Next, let us import all the required keras packages used to build our CNN.

```python
# TASK 3.1.2 Import the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
```
Each of this package corresponds to a layer of the model we are going to build. Before we go into details of each part, let's look at the overall structure of a common CNN model. As shown below, a CNN takes the image, pass it through a series of convolutional, nonlinear, pooling, and fully connected layers, then get an output. Each layer contains many neurons where the computation takes place.

<p align="center">
  <img src="https://github.com/anqitu/NTUOSS-ImageRecognitionWorkshop/blob/master/screenshots/task_3_1_2_a.png">
</p> 
[Source](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/)



To understand the actual math behind all the concepts, i suggest you to go learn from an external source. As this workshop concentrates more on the implementation part, I will only briefly talk about each concept.

Now, let us see what each of the above packages are imported for:

- **Sequential**: The Sequential model is used to initialise our neural network model, so that we can add layers in this model.
- **Conv2D**: The convolution operation is usually the first step of a CNN on the training images. This layer contains many filters each of which can be imagined as a flashlight that is shedding light upon and sliding over the image. The flashlight in this layer is looking for specific features. If they find the features they are looking for, they produce a high activation. Each filter is initialized randomly initially and will be modified during the training process. Since we are working on images here, which a basically 2D arrays, we’re using Convolution 2-D. ([Read more on Convolutional Layers](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/))</br></br>

<p align="center">
  <img src="https://github.com/anqitu/NTUOSS-ImageRecognitionWorkshop/blob/master/screenshots/task_3_1_2_b.gif">
</p> 

</br></br>The above example shows a filter of size 3 * 3, sliding over a image size of 5 * 5. Let's do a quick math: how many different locations in the image can the flashlight shed light upon? The answer is actually shown on the right side of the example, i.e 9.

- **MaxPooling2D**: MaxPooling2D is used for pooling operation. Pooling layer is mostly used immediately after the convolutional layer to reduce the spatial size (only width and height, not depth). This reduces the number of parameters, hence reducing the computation and avoiding overfitting. ([Read more on Pooling Layers](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/))</br></br>

<p align="center">
  <img src="https://github.com/anqitu/NTUOSS-ImageRecognitionWorkshop/blob/master/screenshots/task_3_1_2_c.png">
</p> 

</br></br>The above exmaple shows a max pool layer with filter size 2×2 and stride 2. The output is the max value in a 2×2 region shown using encircled digits. The stride is the pixel step by which we slide the filter. When the stride is 2, the filters jump 2 pixels at a time. This max pool layer reduces the number of parameters by half.

- **Flatten**: Flattening is the process of converting all the resultant 3-D arrays into a single long continuous linear vector.
- **Dense**: We also import Dense to perform the full connection of the neural network. A dense layer is a regular layer of neurons in a neural network. Each neuron receives input from all the neurons in the previous layer, thus densely connected. ([Read more on Neural Network](http://cs231n.github.io/neural-networks-1/))

<p align="center">
  <img src="https://github.com/anqitu/NTUOSS-ImageRecognitionWorkshop/blob/master/screenshots/task_3_1_2_d.png" width="500">
</p> 

#### 3.2 Construct Model

Now, let's build our first model, a simple stack of 3 convolution layers with a ReLU activation and followed by max-pooling layers.

Firstly, initialize the model as a sequential constructor to which we can pass a list of layer instances.
```python
# TASK 3.2.1 Initialize Neural Network Model
model = Sequential()
```

Then, we add the first set of Since CONV -> RELU -> POOL layers. You will be surprised to see how easy it is to actually implement these complex operations in a single line of code in python, thanks to Keras.

```python
# TASK 3.2.2 Create first set of CONV -> RELU -> POOL layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3))) # output shape = (148, 148, 32)
model.add(MaxPooling2D(pool_size=(2, 2))) # output shape = (74, 74, 32)
```

Let’s break down the above code parameter by parameter. We take the sequential model object, then add a convolution layer by using the “Conv2D” function. The Conv2D function is taking 4 arguments:
- the number of filters i.e 32
- the size of each filter i.e 3x3
- the activation function to be used i.e ‘relu’. ReLU (Rectified Linear Units) is an the activation layer that introduces nonlinearity to a system. ([Read more on ReLU](http://www.cs.toronto.edu/~fritz/absps/reluICML.pdf))

- the input shape. We only need to specify the input shape for the first layer. Since, we’re using the TensorFlow backend, we arrange the input shape with “channels last” data ordering. The height and width should also corresponds to the image size we set for the data generator, which is (150, 150, 3).

The output shape after the first convolutional layer will be (148, 148, 32). *Think about why*


Then we have a MaxPooling layer of size (2*2) to reduce the total number of nodes for the upcoming layers. When not specified, the stride is set as the value of the MaxPooling layer size i.e 2. The output shape of this layer will be (74, 74, 32). *Think about why*

Next, we add the second set of CONV -> RELU -> POOL layers

```python
# TASK 3.2.3 Create second set of CONV -> RELU -> POOL layers
model.add(Conv2D(64, (3, 3), activation='relu')) # output shape = (72, 72, 64)
model.add(MaxPooling2D(pool_size=(2, 2))) # output shape = (36, 36, 64)
```
Notice here we do not need to specify the input shape any more.

The output shape after the convolutional layer will be (72, 72, 64). *Think about why*

The output shape after the MaxPooling layer will be (36, 36, 64). *Think about why*

Then, we add the third set of CONV -> RELU -> POOL layers

```python
# TASK 3.2.4 Create third set of CONV -> RELU -> POOL layers
model.add(Conv2D(64, (3, 3), activation='relu')) # output shape = (34, 34, 64)
model.add(MaxPooling2D(pool_size=(2, 2))) # output shape = (17, 17, 64)
```

The output shape after the convolutional layer will be (34, 34, 64). *Think about why*

The output shape after the MaxPooling layer will be (17, 17, 64). *Think about why*

Now, it’s time for us to convert all the pooled images into a continuous vector through Flattening. Flattening takes the 3D array, i.e pooled image pixels, and converts them to a 1D single vector.

```python
# TASK 3.2.5 Convert the 3D feature arrays to 1D vector
model.add(Flatten()) # output shape = (18496,)
```
The output shape after flattening would be (18496,). *Think about why*

After flattening, we need to create a fully connected layer to which we connect the set of nodes we got after the flattening step. As this layer is present between the input layer and output layer, we can refer to it a hidden layer.

```python
# TASK 3.2.6 Add the connection layer
model.add(Dense(units = 256, activation='relu'))
```

Dense is the function to add a fully connected layer, ‘units’ is where we define the number of nodes that should be present in this hidden layer. It always required many experimental tries to choose the most optimal number of nodes.


Then, we add the output layer. The number here indicates the shape of output layer. The output layer should contain only one node, as it is binary classification. A sigmoid activation is perfect for a binary classification as it restricts the results between 0 and 1. This single node will give us a binary output of either a Cat or Dog.

```python
# TASK 3.2.7 Add the output layer
model.add(Dense(units = 1, activation = 'sigmoid'))
```


Lastly, compile the model.

```python
# TASK 3.2.8 Compile the model
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
```
Let us break down the function parameter by parameter again:
- **loss**: The function to compute an error value to represent the difference between the actual output and the predicted output. During our training of the model, we will attempt to either minimize or maximize the value. ```binary_crossentropy``` is usually used for a binary problem. ([Ream more on Cross Entropy](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/))
- **optimizer**: The function used to update the weights and bias (i.e. the internal parameters of a model) in such a way that the error computed by the loss function is minimized. ([Read more on Optimizer](https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c)) ([Read more on RMSProp](https://medium.com/100-days-of-algorithms/day-69-rmsprop-7a88d475003b))
- **metrics**: List of metrics to be evaluated by the model during training and testing. ```accuracy``` means the percentage of correct answers. Metric values are recorded at the end of each epoch on the training dataset.

#### 3.3 Check Model

```python
# TASK 3.3 Check the model structure.
model.summary()
```
You will see the structure of model as below if you follow all steps correctly so far.
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
Finally, it’s time to fit our CNN model to the image dataset that we have preprocessed. As we used a data generator for preparing our training image data, we will use ```.fit_generator``` when fitting the model to the data. Otherwise, ```.fit``` will be used if the training data is preprocessed arrays of number.
```python
# TASK 4: Train Model [WARNING: It took me 84.95 min to complete the training process]
import time
train_start_time = time.time()

from keras.callbacks import EarlyStopping
model.fit_generator(
    generator = train_data,
    epochs = 25,
    callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0)],
    validation_data = validation_data)
model.save('./drive/NTUOSS-ImageRecognitionWorkshop/cnn_model_basic.h5') # save model
print("It takes {:.2f} min to train the model".format((time.time() - train_start_time)/60 ))
```
Again, let us break down the code by parameters:
- **generator**: The data generator we have created previously for training image data.
- **epochs**: The number of steps to train our CNN model. For each single step, the neural network is trained on every training samples to update its parameters (weights and bias).
- **callbacks**: A set of functions to be applied at given stages of the training procedure. Here, we use ```EarlyStopping``` which early stops the training when a monitored quantity has stopped improving. Here, we set the monitored value as ```val_loss```, which is the error value calculated by the loss function for the validation dataset. ```patience``` indicates the number of epochs with no improvement after which training will be stopped. Here, the training will stop early is the 'val_loss' does not improve for two epochs. ```verbose```specifies the verbosity mode: 0 = silent, 1 = progress bar, 2 = one line per epoch.
- **validation_data**: The data generator we have created previously for validation image data.


You will see the model training progress as below.
```
Epoch 1/25
80/80 [==============================] - 3899s 49s/step - loss: 0.7659 - acc: 0.5497 - val_loss: 0.8040 - val_acc: 0.5170
Epoch 2/25
80/80 [==============================] - 74s 930ms/step - loss: 0.6696 - acc: 0.5888 - val_loss: 0.6717 - val_acc: 0.5840
Epoch 3/25
80/80 [==============================] - 76s 945ms/step - loss: 0.6405 - acc: 0.6343 - val_loss: 0.6078 - val_acc: 0.6805
Epoch 4/25
80/80 [==============================] - 78s 976ms/step - loss: 0.6295 - acc: 0.6510 - val_loss: 0.6095 - val_acc: 0.6760
Epoch 5/25
80/80 [==============================] - 77s 963ms/step - loss: 0.6258 - acc: 0.6650 - val_loss: 0.5453 - val_acc: 0.7240
Epoch 6/25
80/80 [==============================] - 74s 926ms/step - loss: 0.6025 - acc: 0.6703 - val_loss: 0.6343 - val_acc: 0.6390
Epoch 7/25
80/80 [==============================] - 74s 925ms/step - loss: 0.5982 - acc: 0.6875 - val_loss: 0.5321 - val_acc: 0.7360
Epoch 8/25
80/80 [==============================] - 73s 918ms/step - loss: 0.5880 - acc: 0.6948 - val_loss: 0.5221 - val_acc: 0.7485
Epoch 9/25
80/80 [==============================] - 75s 936ms/step - loss: 0.5612 - acc: 0.7093 - val_loss: 0.5522 - val_acc: 0.7185
Epoch 10/25
80/80 [==============================] - 74s 919ms/step - loss: 0.5626 - acc: 0.7087 - val_loss: 0.5210 - val_acc: 0.7410
Epoch 11/25
80/80 [==============================] - 73s 916ms/step - loss: 0.5476 - acc: 0.7208 - val_loss: 0.5311 - val_acc: 0.7445
Epoch 12/25
80/80 [==============================] - 74s 923ms/step - loss: 0.5380 - acc: 0.7285 - val_loss: 0.5119 - val_acc: 0.7515
Epoch 13/25
80/80 [==============================] - 73s 917ms/step - loss: 0.5444 - acc: 0.7278 - val_loss: 0.5058 - val_acc: 0.7645
Epoch 14/25
80/80 [==============================] - 75s 933ms/step - loss: 0.5301 - acc: 0.7298 - val_loss: 0.4769 - val_acc: 0.7860
Epoch 15/25
80/80 [==============================] - 73s 918ms/step - loss: 0.5304 - acc: 0.7280 - val_loss: 0.4607 - val_acc: 0.7900
Epoch 16/25
80/80 [==============================] - 73s 918ms/step - loss: 0.5227 - acc: 0.7420 - val_loss: 0.5298 - val_acc: 0.7335
Epoch 17/25
80/80 [==============================] - 74s 922ms/step - loss: 0.5204 - acc: 0.7395 - val_loss: 0.4897 - val_acc: 0.7775
It takes 84.95 min to train the model
```

For your information, it took me 84.95 min to train the model (as the there are lots of computations involved). Thus, I suggest you to use the trained model inside the model folder for the rest of the workshop, and try training your own model when you go back home.

## Task 5 - Test Model
Sa we can see from the training history, the final accuracy score is 74% for train data and 78% for validation data. Now let us test the model with our own test data.

#### 5.1 Load Model

Firstly, we need to load the model that I have trained before this workshop with the codes above. All we need to do is to specify the location of the model in the ```load_model``` function.
```python
# TASK 5.1: Load trained model
from keras.models import load_model
model_basic = load_model('./drive/NTUOSS-ImageRecognitionWorkshop/data/model/cnn_model_basic.h5')
```

#### 5.2 Generate Test Image Data from Directory
Similar to how we generate data for train and validation data, we can also do the same for test data. The setting for test data generator should have the same target_size as train generator settings, but different directory path, class mode, shuffle choice and batch_size.
- **rescale**: For the test set, only rescale should be made because we don't want to mess with new data and just predict its class.
- **directory**: The directory should be set to as the test data folder.
- **target_size = (150, 150)**: The image size is same as that used for training data.
- **class_mode = None**: Set this to None, to return only the images.
- **shuffle = False,**: Set this to False, because we will need to yield the images in order, to predict the outputs and match them with their filenames.
- **batch_size = 200**: For train and validation data, it was set to be 50 which divides 2000. For test data, set this to some number that divides your total number of images in your test set exactly. Here we can set it to 200 (or any number then divides the total number of test images).

```python
# TASK 5.2.1: Set up data generator for test data
from keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_data = datagen_test.flow_from_directory (
    directory = './drive/NTUOSS-ImageRecognitionWorkshop/data/test',
    target_size = (150, 150),
    class_mode = None,
    shuffle = False,
    batch_size = 200)
```

And you will see the response as below:
```
Found 200 images belonging to 2 classes.
```

Then, check the class indices and filenames.
```python
# TASK 5.2.2: Check test generator
print(test_data.class_indices)
print(test_data.filenames)
```

As shown below, the filenames help indicate the class of the image.
```
{'cat': 0, 'dog': 1}
['cat/cat.1029.jpg', 'cat/cat.10603.jpg', 'cat/cat.10629.jpg', ...'dog/dog.756.jpg', 'dog/dog.779.jpg', 'dog/dog.87.jpg']
```


#### 5.3 Make Prediction
Then, we use the ```.predict_generator``` function of the model to classify out test data. Simply put the test data in the parameter.
```python
# TASK 5.3.1: Use model to yield probability prediction for test data
probabilities = model_basic.predict_generator(test_data)
print(probabilities)
```
Here, we see the output for each image is a probability between 0 and 1.
```
[[0.14854874]
 [0.74102986]
 [0.64901006]
 ...
 ...
 [0.9367852 ]
 [0.9622177 ]
 [0.607676  ]]
```

Here, we set the threshold as 0.5, which means a probability above 0.5 indicates 1 (dog), otherwise 0 (cat).
```python
# TASK 5.3.2: Process probabilities to get prediction result
y_pred = [1 if prob > 0.5 else 0 for prob in probabilities]
print(y_pred)
```

```
[0, 1, 1, 0, ... 0, 1, 1, 1]
```

Then, we need to prepare the actual list of class result using the class referred from filenames.
```python
# TASK 5.3.3: Prepare actual result using filenames
y_true = [0 if 'cat' in filename else 1 for filename in test_data.filenames]
print(y_true)
```

```
[0, 0, 0, 0,... 1, 1, 1, 1]
```

Now, we have the predicted result by the model and the true result. We can use ```accuracy_score``` function from [Sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) to calculate the accuracy score. The function takes in two arguments - the predicted result and the true result.
```python
# TASK 5.3.4: Calculate accuracy score
from sklearn.metrics import accuracy_score
print(accuracy_score(y_true, y_pred))
```

The accuracy score of the predicted result by our basic model is 77%. This is quite impressive, considering the size of our training data and the complexity of our CNN model.
```
0.77
```

We can also generate a more detailed report with [Pandas](https://pandas.pydata.org/pandas-docs/version/0.22.0/generated/pandas.crosstab.html).
```python
# TASK 5.3.5: Generate a report
import pandas as pd
pd.crosstab(pd.Series(y_true), pd.Series(y_pred), rownames=['True'], colnames=['Pred'], margins=True)
```

Here, we can see that our model predict 65 out of 100 cat images and 89 out of dog images correctly.
```
Pred	0	1	All
True			
0	65	35	100
1	11	89	100
All	76	124	200
```


## Task 6 - Recognize Image with Basic Model
Now, we come to the most second exciting part of this workshop - use our model to recognize and classify an image!

#### 6.1 Useful Functions

Before that, we need to code some useful functions to read the image, preprocess the image and process the predicting result.

This first function returns an image from a given url.
```python
# TASK 6.1.1: Define a function for reading image from url
def read_image_from_url(url):
    try:
        import requests, io
        from PIL import Image
        r = requests.get(url, timeout=15)
        img = Image.open(io.BytesIO(r.content))
        return img
    except:
        print("{:<10} Cannot find image from {}".format('[ERROR]', url))
        exit(1)
```

This second function process an given image to an arrays in the format that match our model's input requirements.
```python
# TASK 6.1.2: Define a function for preprocessing image
def preprocess_image(img, target_size):
    from PIL import Image
    import numpy as np
    from keras.preprocessing.image import img_to_array
    img = img.resize(target_size,Image.ANTIALIAS)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')
    img /= 255
    return img

```

This last function returns whether a 'dog' or a 'cat' given the probability outputted by the model
```python
# TASK 6.1.3: Define a function for processing result
def process_result(prob):
    return 'dog' if prob > 0.5 else 'cat'
```

#### 6.2 Make Prediction
Now, you can play with the model and feel free to copy paste the url of any dog or cat image to replace the url below.
```python
# TASK 6.2: Read image, Preprocess image, and Make prediction
image = read_image_from_url('https://www.readersdigest.ca/wp-content/uploads/2011/01/4-ways-cheer-up-depressed-cat.jpg') # replace with any image url
image = preprocess_image(image, (150, 150))
prob = model_basic.predict(image)
print('Probability: ' + str(prob))
print('Class: ' + process_result(prob))
```

There it goes, the model predicts the image as a cat accurately!
```
Probability: [[0.3044719]]
Class: cat
```
So far, we have successfully built from scratch a basic CNN model, which has a acceptable level of accuracy. The question now is - is there any way for us to push for a higher accuracy yet without any much more efforts?

## Task 7 - Build on Top of Pretrained Model

To train a CNN from scratch in order to achieve high accuracy, it takes many images (on the order of hundreds of thousands). For example, [VGG16](https://arxiv.org/abs/1409.1556) is a convolutional neural network model trained on 14 million images to recognize an image as one of 1000 categories with an accuracy of 92.5%.

A more refined approach is to leverage a CNN that is already pre-trained on a large dataset. Such a network would have already learned features that are useful for most computer vision problems. Leveraging such features would allow us to reach a higher accuracy in a faster manner.

For this workshop, we will use the VGG16 CNN model. Because its dataset contains several "cat" classes and many "dog" classes among its total of 1000 classes, this model will already have learned features that are relevant to our classification problem.


#### 7.1 Set up backend and Import libraries

First of all, configure the backend.

```python
# TASK 7.1.1 Configure backend
from keras import backend as K
K.set_image_dim_ordering('tf') #channel last
K.set_image_data_format('channels_last')
```

Then, import necessary Keras libraries and packages. Notice that here we add a Dropout layer whose purpose is to avoid overfitting with regularization.

```python
# TASK 7.1.2 Import the Keras libraries and packages
from keras.applications import VGG16
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
```

#### 7.2 Construct Model
To construct the mode, let us first load the VGG model.
```python
# TASK 7.2.1: Load base model
base_model = VGG16(include_top = False, weights = 'imagenet', input_shape = (150, 150, 3))
```
We break down the code by paramaters again:
- **include_top = False**: whether to include the 3 fully-connected layers at the top of the network.
- **weights = 'imagenet'**: weights to initialize the model with. Here we set it as the weights that have been pre-trained on ImageNet dataset.
- **input_shape = (150, 150, 3)**: image shape of our image data.

Then, we add new dense and output layers on top of the base model. Different from the previous basic model, we add a dropout layer here. The dropout randomly drops 40% of connections of neurons from the dense layer to prevent overfitting.

```python
# TASK 7.2.2: Add new layers
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x) #new FC layer, random init
x = Dropout(0.4)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
```

Now our model has two part - base model and new layers. Since the base model contains layers that have already learnt useful patterns, we do not want to retrain those payers during our training. Thus, let us set those layers as untrainable and compile the model.
```python
# TASK 7.2.3: Setup trainable layer
for layer in base_model.layers:
    layer.trainable = False
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
print("{:<10} Pretrained model layers: {}".format('[INFO]', len(base_model.layers)))
print("{:<10} Total number of layers : {}".format('[INFO]', len(model.layers)))
```

Here, we can see that 19 out of the 23 layers are not trainable.
```
[INFO]     Pretrained model layers: 19
[INFO]     Total number of layers : 23
```
Now, compile the model with the same settings as the previous basic model.
```python
# TASK 7.2.4: Compile model
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
```

#### 7.3 Check Model

Let us check the structure of the advanced model.
```python
# TASK 7.3: Check Model
model.summary()
```

It is obviously a much more complicated model than the one we built before.
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 150, 150, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 150, 150, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 150, 150, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 75, 75, 64)        0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 75, 75, 128)       73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 75, 75, 128)       147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 37, 37, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 37, 37, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 37, 37, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 37, 37, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 18, 18, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 18, 18, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 18, 18, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 18, 18, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 9, 9, 512)         0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0         
_________________________________________________________________
flatten_3 (Flatten)          (None, 8192)              0         
_________________________________________________________________
dense_5 (Dense)              (None, 256)               2097408   
_________________________________________________________________
dropout_2 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_6 (Dense)              (None, 1)                 257       
=================================================================
Total params: 16,812,353
Trainable params: 2,097,665
Non-trainable params: 14,714,688
_________________________________________________________________
```

## Task 8 - Train Advanced Model
Usually, the larger the model, the longer the time taken to train the model. However, as we already set most of the layers untrainable, the model will actually take less time to train compared to the previous basic model.

The training code is same as the one used for the basic model.
```python
# TASK 8: Train advanced model.
import time
train_start_time = time.time()

from keras.callbacks import EarlyStopping

model.fit_generator(
    train_data,
    epochs = 25,
    callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0)],
    validation_data = validation_data)
model.save('./drive/NTUOSS-ImageRecognitionWorkshop/cnn_model_advanced.h5')

print("It takes {:.2f} min to train the model".format((time.time() - train_start_time)/60 ))
```

We can observe that the model takes much less time to train yet achieves higher accuracy! This means our strategy works!
```
Epoch 1/25
80/80 [==============================] - 88s 1s/step - loss: 0.8231 - acc: 0.6978 - val_loss: 0.3416 - val_acc: 0.8390
Epoch 2/25
80/80 [==============================] - 79s 984ms/step - loss: 0.4639 - acc: 0.7767 - val_loss: 0.3518 - val_acc: 0.8345
Epoch 3/25
80/80 [==============================] - 79s 993ms/step - loss: 0.4175 - acc: 0.8007 - val_loss: 0.2483 - val_acc: 0.8980
Epoch 4/25
80/80 [==============================] - 79s 982ms/step - loss: 0.3830 - acc: 0.8240 - val_loss: 0.2786 - val_acc: 0.8740
Epoch 5/25
80/80 [==============================] - 79s 981ms/step - loss: 0.3834 - acc: 0.8265 - val_loss: 0.2862 - val_acc: 0.8760
It takes 7.12 min to train the model
```

## Task 9 - Test Advanced Model
Same as before, we load the advanced model that I trained before this workshop.

#### 9.1 Load Model
Put the address of the model in the parameter of the ```.load_model``` function.
```python
# TASK 9.1: Load trained model
from keras.models import load_model
model_advanced = load_model('./drive/NTUOSS-ImageRecognitionWorkshop/data/model/cnn_model_advanced.h5')
```

#### 9.2 Generate Test Image Data from Directory
Then, generate test data from directory with the same settings.
```python
# TASK 9.2.1 : Set up data generator for test data
from keras.preprocessing.image import ImageDataGenerator
datagen_test = ImageDataGenerator(rescale=1. / 255)

test_data = datagen_test.flow_from_directory (
    directory = './drive/NTUOSS-ImageRecognitionWorkshop/data/test',
    target_size = (150, 150),
    class_mode = None,
    shuffle = False, # keep data in same order as labels
    batch_size = 200)
```

Check the test data generator.
```python
# TASK 9.2.2: Check test data generator
print(test_data.class_indices)
print(test_data.filenames)
```

#### 9.3 Make Prediction

Use the advanced model to make predictions for test data.
```python
# TASK 9.3.1: Use model to yield probability prediction for test data
probabilities = model_advanced.predict_generator(test_data)
print(probabilities)
```

```
[[8.9933887e-02]
 [5.1956624e-01]
 [5.7828718e-01]
 ...
 ...
 [9.9898058e-01]
 [9.9089313e-01]
 [9.9960285e-01]]
```

Process probabilities to get prediction result.
```python
# TASK 9.3.2: Process probabilities to get prediction result
y_pred = [1 if prob > 0.5 else 0 for prob in probabilities]
print(y_pred)
```

```
[0, 1, 1, 0, ... 1, 1, 1, 1]
```

Prepare the actual results using folder name in filenames.
```python
# TASK 9.3.3: Prepare actual result using folder name in filenames
y_true = [0 if 'cat' in filename else 1 for filename in test_data.filenames]
print(y_true)
```

```
[0, 0, 0, 0, 0, ... 1, 1, 1, 1, 1]
```

Calculate accuracy score.
```python
# TASK 9.3.4: Calculate accuracy score
from sklearn.metrics import accuracy_score
print(accuracy_score(y_true, y_pred))
```

We can observe a higher accuracy score.
```
0.83
```

Then, generate a more detailed report.
```python
# TASK 9.3.5: Generate a report
import pandas as pd
pd.crosstab(pd.Series(y_true), pd.Series(y_pred), rownames = ['True'], colnames = ['Pred'], margins = True)
```

Are you able to read the report below?
```
Pred	0	1	All
True			
0	70	30	100
1	4	96	100
All	74	126	200
```


## Task 10 - Recognize Image with Advanced Model

Now, feel free to play with the advanced model and have fun!
#### 10.1 Make Prediction

```python
# TASK 10.1: Read image, Preprocess image, and Make prediction
image = read_image_from_url('https://www.readersdigest.ca/wp-content/uploads/2011/01/4-ways-cheer-up-depressed-cat.jpg') # replace with any image url
image = preprocess_image(image, (150, 150))
prob = model_advanced.predict(image)
print('Probability: ' + str(prob))
print('Class: ' + process_result(prob))
```

```
Probability: [[0.18195716]]
Class: cat
```

I hope after this workshop, all of you are able to build your own CNN model with Keras. There are also many ways to improve this model, such as adding more training data, adding more layers, building broader networks, modify parameters and so on. Again, this workshop focuses more on the implementation part. If you are interested in how the underlying math works or what are the most advanced CNN architecture, do source for any external resources.
___

## Acknowledgements

Many thanks to [clarencecastillo](https://github.com/clarencecastillo) for carefully testing this walkthrough and to everybody else in [NTU Open Source Society](https://github.com/ntuoss) committee for making this happen! :kissing_heart::kissing_heart::kissing_heart:

## Resources
[Keras Docs](https://keras.io/)</br>
[VGG16 Docs](https://arxiv.org/abs/1409.1556)</br>
[Sklearn Docs](http://scikit-learn.org/)</br>
[A Beginner's Guide To Understanding CNNs](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/)</br>
[Image Classification using CNNs in Keras](https://www.learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/)</br>
[CS231n CNNS for Visual Recognition](http://cs231n.github.io/convolutional-networks/)</br>
[Simple Image Classification using CNN](https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8)
