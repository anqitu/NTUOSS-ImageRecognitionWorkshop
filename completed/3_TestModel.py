# Settings
import os
project_path = '/Users/anqitu/Workspaces/OSS/NTUOSS-ImageRecognitionWorkshop'
data_path = os.path.join(project_path, 'data')
image_path_test = os.path.join(data_path, 'test')
model_path = os.path.join(project_path, 'completed/model')

## Task 5 - Test Model
BATCH_SIZE = 32
IM_WIDTH, IM_HEIGHT = 150, 150

from keras.preprocessing.image import ImageDataGenerator
test_datagen =  ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory (
        image_path_test,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BATCH_SIZE,
        shuffle=False)  # keep data in same order as labels

test_generator.class_indices
test_generator.filenames
y_true = [0 if 'cat' in filename else 1 for filename in test_generator.filenames]

from keras.models import load_model
model = load_model(os.path.join(model_path, 'cnn_model_advanced.h5'))

probabilities = model.predict_generator(test_generator)

from sklearn.metrics import confusion_matrix
y_pred = probabilities > 0.5

confusion_matrix(y_true, y_pred)
