# Settings
import os
project_path = '/Users/anqitu/Workspaces/OSS/NTUOSS-ImageRecognicationWorkshop'
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
        class_mode=None,  # only data, no labels
        shuffle=False)  # keep data in same order as labels

from keras.models import load_model
model = load_model(os.path.join(model_path, 'cnn_model.h5'))


probabilities = model.predict_generator(generator, 2000)


from sklearn.metrics import confusion_matrix

y_true = np.array([0] * 1000 + [1] * 1000)
y_pred = probabilities > 0.5

confusion_matrix(y_true, y_pred)
