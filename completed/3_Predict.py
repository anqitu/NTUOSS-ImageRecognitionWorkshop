import os

# Settings
project_path = '/Users/anqitu/Workspaces/OSS/NTUOSS-ImageRecognicationWorkshop'
data_path = os.path.join(project_path, 'data')
image_path_train = os.path.join(data_path, 'train')
image_path_val = os.path.join(data_path, 'validation')
image_path_test = os.path.join(data_path, 'test')
model_path = os.path.join(project_path, 'completed/model/cnn_model.h5')


SEED = 0
BATCH_SIZE = 32
EPOCHS = 25
CLASSES = ['loaf bread', 'corgi butt']
CLASS_SIZE = len(CLASSES)
IM_WIDTH, IM_HEIGHT = 150, 150

from keras.preprocessing.image import ImageDataGenerator
test_datagen =  ImageDataGenerator(rescale=1. / 255)
generator = datagen.flow_from_directory (
        image_path_test,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode=None,  # only data, no labels
        shuffle=False)  # keep data in same order as labels

probabilities = model.predict_generator(generator, 2000)


from sklearn.metrics import confusion_matrix

y_true = np.array([0] * 1000 + [1] * 1000)
y_pred = probabilities > 0.5

confusion_matrix(y_true, y_pred)
