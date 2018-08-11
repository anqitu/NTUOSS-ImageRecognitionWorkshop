import time
import os

import warnings
warnings.filterwarnings('ignore')






# Settings
project_path = '/Users/anqitu/Workspaces/OSS/NTUOSS-ImageRecognicationWorkshop/'
data_path = os.path.join(project_path, 'DataFile')
image_path_train = os.path.join(data_path, 'ImagesTrain')
image_path_val = os.path.join(data_path, 'ImagesVal')
model_path = os.path.join(project_path, 'Model')

SEED = 0
BATCH_SIZE = 32
EPOCHS = 25
CLASSES = ['loaf bread', 'corgi butt']
CLASS_SIZE = len(classes)
IM_WIDTH, IM_HEIGHT = 150, 150
TRAIN_SIZE = len([os.path.join(root, name) for root, dirs, files in os.walk(image_path_train) for name in files if name!= '.DS_Store'])
VAL_SIZE = len([os.path.join(root, name) for root, dirs, files in os.walk(image_path_val) for name in files if name!= '.DS_Store'])



# Build model ------------------------------------------------------------------
# The code snippet below is our first model, a simple stack of 3 convolution
# layers with a ReLU activation and followed by max-pooling layers.
from keras import backend as K
K.set_image_data_format('channels_last')
K.set_image_dim_ordering('tf') #channel last

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(IM_WIDTH, IM_HEIGHT, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard

train_datagen =  ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
val_datagen = ImageDataGenerator(
    rescale=1. / 255)
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

train_generator.class_indices



train_start_time = time.time()
model.fit_generator(
    train_generator,
    steps_per_epoch= TRAIN_SIZE // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0), TensorBoard(log_dir="TensorBoard/logs/{}".format(time.time()))],
    validation_data = val_generator,
    validation_steps= VAL_SIZE // BATCH_SIZE)
model.save(os.path.join(model_path, 'cnn_mdel'+'.h5'))
print("It takes {:.2f} min to train the model".format((time.time() - train_start_time)/60 ))

# tensorboard --logdir=/Users/anqitu/Workspaces/OSS/NTUOSS-ImageRecognicationWorkshop/TensorBoard/logs
# localhost:6006
