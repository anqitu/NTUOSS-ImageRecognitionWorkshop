# Settings
import os
project_path = '/Users/anqitu/Workspaces/OSS/NTUOSS-ImageRecognitionWorkshop'
data_path = os.path.join(project_path, 'data')
image_path_train = os.path.join(data_path, 'train')
image_path_val = os.path.join(data_path, 'validation')
model_path = os.path.join(project_path, 'completed/model/')

## Task 2 - Preprocess Images
IM_WIDTH, IM_HEIGHT = 150, 150
BATCH_SIZE = 32

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


## Task 3 - Build Basic Model
# VGG-like convnet from Keras: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
from keras import backend as K
K.set_image_data_format('channels_last')
K.set_image_dim_ordering('tf') #channel last

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IM_WIDTH, IM_HEIGHT, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts the 3D feature maps to 1D feature vectors
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.layer

## Task 4 - Train Model
EPOCHS = 25

from keras.callbacks import EarlyStopping
train_start_time = time.time()
model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0)],
    validation_data = val_generator)
model.save(os.path.join(model_path, 'cnn_model.h5'))
print("It takes {:.2f} min to train the model".format((time.time() - train_start_time)/60 ))







## Task 7 - Build Advanced Model
IM_WIDTH, IM_HEIGHT = 150, 150

from keras.applications import VGG16
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model

# add new layer
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(IM_WIDTH, IM_WIDTH, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x) #new FC layer, random init
x = Dropout(0.4)(x)
predictions = Dense(1, activation='sigmoid')(x) #new softmax layer
model = Model(inputs=base_model.input, outputs=predictions)

model.summary()

# setup to transfer learn
for layer in base_model.layers:
    layer.trainable = False
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print("{:<10} Pretrained model layers: {}".format('[INFO]', len(base_model.layers)))
print("{:<10} New model layers       : {}".format('[INFO]', len(model.layers)))


## Task 4 - Train Model
EPOCHS = 25

from keras.callbacks import EarlyStopping
import time
train_start_time = time.time()
model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0)],
    validation_data = val_generator)
model.save(os.path.join(model_path, 'cnn_model_advanced.h5'))
print("It takes {:.2f} min to train the model".format((time.time() - train_start_time)/60 ))
