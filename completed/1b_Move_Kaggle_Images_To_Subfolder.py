# Download image data from Kaggle: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data.
# Move it inside the working directory folder: '/Users/anqitu/Workspaces/OSS/NTUOSS-ImageRecognitionWorkshop/cats_vs_dogs'

import os
import glob
import pandas as pd

directory_origin = '/Users/anqitu/Workspaces/OSS/NTUOSS-ImageRecognitionWorkshop/cats_vs_dogs'
directory_data = '/Users/anqitu/Workspaces/OSS/NTUOSS-ImageRecognitionWorkshop/data'

def get_sub_fpaths(folder_path):
	import os
	return [os.path.join(folder_path,f) for f in next(os.walk(folder_path))[2] if f!= '.DS_Store']

image_paths = get_sub_fpaths(directory_origin)
image_paths[0]


image_paths_dog = [image_path for image_path in image_paths if 'dog.' in image_path]
image_paths_cat = [image_path for image_path in image_paths if 'cat.' in image_path]
len(image_paths_dog)
len(image_paths_cat)

def check_dir(directory):
    if not os.path.exists(directory):
        check_dir(os.path.dirname(directory))
        os.mkdir(directory)
        print("{:<6} Make directory: {}".format('[INFO]', directory))

for data_type in ['train', 'test', 'validation']:
    for label in ['cat', 'dog']:
        check_dir(os.path.join(directory_data, data_type, label))

from sklearn.model_selection import train_test_split
others, train_paths = train_test_split(image_paths_dog, test_size=2000, random_state=2018)
others, validation_paths = train_test_split(others, test_size=1000, random_state=2018)
others, test_paths = train_test_split(others, test_size=500, random_state=2018)

len(train_paths)
len(validation_paths)
len(test_paths)

for image_path in train_paths:
    os.rename(image_path, os.path.join(directory_data, 'train', 'dog', os.path.basename(image_path)))

for image_path in validation_paths:
    os.rename(image_path, os.path.join(directory_data, 'validation', 'dog', os.path.basename(image_path)))

for image_path in test_paths:
    os.rename(image_path, os.path.join(directory_data, 'test', 'dog', os.path.basename(image_path)))


from sklearn.model_selection import train_test_split
others, train_paths = train_test_split(image_paths_cat, test_size=2000, random_state=2018)
others, validation_paths = train_test_split(others, test_size=1000, random_state=2018)
others, test_paths = train_test_split(others, test_size=500, random_state=2018)

len(train_paths)
len(validation_paths)
len(test_paths)

for image_path in train_paths:
    os.rename(image_path, os.path.join(directory_data, 'train', 'cat', os.path.basename(image_path)))

for image_path in validation_paths:
    os.rename(image_path, os.path.join(directory_data, 'validation', 'cat', os.path.basename(image_path)))

for image_path in test_paths:
    os.rename(image_path, os.path.join(directory_data, 'test', 'cat', os.path.basename(image_path)))
