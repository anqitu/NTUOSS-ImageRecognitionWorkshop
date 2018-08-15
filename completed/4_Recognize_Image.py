## Task 6 - Recognize Image

# Predict images ---------------------------------------------------------------
def read_image_from_path(image_path, target_size = None):
    try:
        from PIL import Image
        img = Image.open(image_path)
        if target_size != None: img = img.resize(target_size, Image.ANTIALIAS)
        return img
    except:
        print("{:<10} Cannot find image from {}".format('[ERROR]', image_path))
        exit(1)

def read_image_from_url(url, target_size = None):
    try:
        import requests, io
        from PIL import Image
        r = requests.get(url, timeout=15)
        img = Image.open(io.BytesIO(r.content))
        if target_size != None: img = img.resize(target_size, Image.ANTIALIAS)
        return img
    except:
        print("{:<10} Cannot find image from {}".format('[ERROR]', url))
        exit(1)

def preprocess_image(img, target_size):
    from PIL import Image
    import numpy as np
    from keras.preprocessing.image import img_to_array
    img = img.resize(target_size,Image.ANTIALIAS)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img  = img.astype('float32')
    img /= 255
    return img

def process_result(predict_class):
    return 'dog' if predict_class == 0 else 'cat'

# Settings
import os
project_path = '/Users/anqitu/Workspaces/OSS/NTUOSS-ImageRecognicationWorkshop'
data_path = os.path.join(project_path, 'data')
image_path_test = os.path.join(data_path, 'test')
model_path = os.path.join(project_path, 'completed/model')

IM_WIDTH, IM_HEIGHT = 150, 150

from keras.models import load_model
model = load_model(os.path.join(model_path, 'cnn_model.h5'))


image = read_image_from_url('https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg')
image = preprocess_image(image, (IM_WIDTH, IM_HEIGHT))
predict_class = model.predict_classes(image)
model.predict(image)
process_result(predict_class)
