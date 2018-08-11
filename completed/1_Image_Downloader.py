# import the necessary packages
from imutils import paths
import argparse
import requests
import cv2
import os, sys, io
from PIL import Image
from util import *
import numpy as np

import warnings
warnings.filterwarnings('ignore')

def download_images(url_fpath, image_requested):
	get_now()
	cate_folder_train = os.path.join(image_path_train, get_fname_from_path(url_fpath))
	cate_folder_val = os.path.join(image_path_val, get_fname_from_path(url_fpath))
	ensure_directory(cate_folder_train)
	ensure_directory(cate_folder_val)

	urls = get_urls(url_fpath, image_requested)
	urls_sum = urls.shape[0]
	train_sum = int(urls_sum * 0.7)

	count = 0
	print("{:<10} Start downloading images for train for {}".format('[INFO]', get_fname_from_path(url_fpath).upper()))
	for url in urls[:train_sum]:
		try:
			r = requests.get(url, timeout=15)
			image = Image.open(io.BytesIO(r.content))
			image_rgb = image.convert("RGB")

			path = os.path.join(cate_folder_train, "{}.jpg".format(str(count).zfill(5)))
			image_rgb.save(path, format='JPEG', quality=85)

			count += 1
		except:
			urls_sum -= 1
			print("{:<10} error downloading ...skipping".format('[INFO]'))

		if count % 50 == 0:
			print("{:<10} downloaded: {}/{}".format('[INFO]',count, urls_sum))
	train_count = count
	print("{:<10} Downloaded {} images for train".format('[INFO]', train_count))

	print("{:<10} Start downloading images for validation for {}".format('[INFO]', get_fname_from_path(url_fpath).upper()))
	for url in urls[train_sum:]:
		try:
			r = requests.get(url, timeout=15)
			image = Image.open(io.BytesIO(r.content))
			image_rgb = image.convert("RGB")

			path = os.path.join(cate_folder_val, "{}.jpg".format(str(count).zfill(5)))
			image_rgb.save(path, format='JPEG', quality=90)

			count += 1
		except:
			urls_sum -= 1
			print("{:<10} error downloading ...skipping".format('[INFO]'))

		if count % 50 == 0:
			print("{:<10} downloaded: {}/{}".format('[INFO]',count, urls_sum))
	print("{:<10} Downloaded {} images for validation".format('[INFO]', count - train_count))


def get_urls(url_path, url_requested):
	urls = np.load(url_path)
	nrow = urls.shape[0]
	if nrow < url_requested:
		print("{:<10} Not enough urls for {}: {} is requested but {} detected'".format('[WARNING]', get_fname_from_path(url_path), url_requested, nrow ))
	return urls[:url_requested]


# loop over the image paths we just downloaded
def check_image(folder):
	for imagePath in get_sub_fpaths(folder):
		delete = False
		try:
			image = cv2.imread(imagePath)
			if image is None:
				delete = True
		except:
			delete = True

		# check to see if the image should be deleted
		if delete:
			print("[INFO] deleting {} from {}".format(get_fname_from_path(imagePath), get_fname_from_path(folder)))
			os.remove(imagePath)

# Get project path
project_path = os.getcwd()
data_path = os.path.join(project_path, 'DataFile')
url_data_path = os.path.join(data_path, 'urls')
image_path_train = os.path.join(data_path, 'ImagesTrain')
image_path_val = os.path.join(data_path, 'ImagesVal')

# Set limit for data size. Larger train size will get higher accuracy, but occupy more storagy and slow down the training speed.
N_URL_LIMIT_LOW = 100 # Recommended
N_URL_LIMIT_HIGH= 500 # Recommended
# N_URL_LIMIT_LOW = 5 # For testing
# N_URL_LIMIT_HIGH= 10 # For testing

classes = [get_fname_from_path(f) for f in get_sub_fnames(url_data_path)]
print("{:<10} Numbe of Classes detected: {}".format('[INFO]', len(classes)))
print("{:<10} Classes: {}".format('', str(classes)))


if __name__ == '__main__':
	argument_parser = argparse.ArgumentParser(description='Download images from urls provided')
	argument_parser.add_argument('--url_fpath', default='all', type=str, help='The url file path to download images from. Download "all" by default')
	argument_parser.add_argument('--count', default=100, type=int, help='How many images to download. 100 by default')
	args = argument_parser.parse_args()

	for path in [image_path_train, image_path_val]:
		ensure_directory(path)

	if args.count < N_URL_LIMIT_LOW:
		print("{:<10} Not enough urls for model training. {} insead of {} will be requested".format('[WARNING]', N_URL_LIMIT_LOW, args.count))
		print("{:<10} You can change the N_URL_LIMIT_LOW setting".format(''))
		args.count = N_URL_LIMIT_LOW

	if args.count > N_URL_LIMIT_HIGH:
		print("{:<10} Too many urls which will slow down the model training. {} insead of {} will be requested".format('[WARNING]', N_URL_LIMIT_HIGH, args.count))
		print("{:<10} You can change the N_URL_LIMIT_HIGH setting".format(''))
		args.count = N_URL_LIMIT_HIGH

	if args.url_fpath == 'all':
		print("{:<10} Downloading from all url paths".format('[INFO]'))
		url_fpaths = get_sub_fpaths(url_data_path)
		for url_path in url_fpaths:
			download_images(url_path, args.count)
	else:
		download_images(args.url_fpath, args.count)

	print("{:<10} Start checking images".format('[INFO]'))
	for image_path in [image_path_train, image_path_val]:
		for cate_folder in get_subfolder_paths(image_path):
			check_image(cate_folder)
	print("{:<10} Finished checking images".format('[INFO]'))
	sys.exit(1)
