# import the necessary packages
import os
import sys
import urllib
import cv2
from queue import Queue
from threading import Thread
from time import strftime
import pandas as pd

from util import *
import numpy as np
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


def download(url, dest):
	try:
		req = urllib.request.Request(url)
		response = urllib.request.urlopen(req, timeout = 10)
		content = response.read()
		with open(dest, 'wb') as f:
			f.write(content)
	except Exception as e:
		print(e, url)

# Thread pool code from Chris Hager's blog
# reference: https://www.metachris.com/2016/04/python-threadpool/
class Worker(Thread):
	""" Thread executing tasks from a given tasks queue """
	def __init__(self, tasks):
		Thread.__init__(self)
		self.tasks = tasks
		self.daemon = True
		self.start()

	def run(self):
		while True:
			func, args, kargs = self.tasks.get()
			try:
				func(*args, **kargs)
			except Exception as e:
				# An exception happened in this thread
				print(e)
			finally:
				# Mark this task as done, whether an exception happened or not
				self.tasks.task_done()

class ThreadPool:
	""" Pool of threads consuming tasks from a queue """
	def __init__(self, num_threads):
		self.tasks = Queue(num_threads)
		for _ in range(num_threads):
			Worker(self.tasks)

	def add_task(self, func, *args, **kargs):
		""" Add a task to the queue """
		self.tasks.put((func, args, kargs))

	def map(self, func, args_list):
		""" Add a list of tasks to the queue """
		for args in args_list:
			self.add_task(func, *args)

	def wait_completion(self):
		""" Wait for completion of all the tasks in the queue """
		self.tasks.join()

# loop over the image paths we just downloaded to ensure the image is in the correct format
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
project_path = '/Users/anqitu/Workspaces/OSS/NTUOSS-ImageRecognitionWorkshop'
data_path = os.path.join(project_path, 'data')
url_data_path = os.path.join(data_path, 'urls')
image_path_train = os.path.join(data_path, 'train')
image_path_val = os.path.join(data_path, 'validation')
image_path_test = os.path.join(data_path, 'test')

for path in [image_path_train, image_path_val, image_path_test]:
	ensure_directory(path)
url_data_path
classes = [get_fname_from_path(f) for f in get_sub_fnames(url_data_path)]
print("{:<10} Numbe of Classes detected: {}".format('[INFO]', len(classes)))
print("{:<10} Classes: {}".format('', str(classes)))



tasks = []

url_paths = get_sub_fpaths(url_data_path)
for url_path in url_paths:
	urls = np.load(url_path)
	train, test = train_test_split(urls, test_size=0.1, random_state=2018)
	train, validation = train_test_split(train, test_size=0.3, random_state=2018)
	category = os.path.basename(url_path).split('.')[0]

	for url in train:
		dst_image_folder = os.path.join(image_path_train, category)
		ensure_directory(dst_image_folder)
		file_name = '{}.{}'.format(str(hash(url)), url.split('.')[-1])
		dst_image_path = os.path.join(dst_image_folder, file_name)
		if not os.path.exists(dst_image_path):
			tasks.append([url, dst_image_path])

	for url in validation:
		dst_image_folder = os.path.join(image_path_val, category)
		ensure_directory(dst_image_folder)
		file_name = '{}.{}'.format(str(hash(url)), url.split('.')[-1])
		dst_image_path = os.path.join(dst_image_folder, file_name)
		if not os.path.exists(dst_image_path):
			tasks.append([url, dst_image_path])

	for url in test:
		dst_image_folder = os.path.join(image_path_test, category)
		ensure_directory(dst_image_folder)
		file_name = '{}.{}'.format(str(hash(url)), url.split('.')[-1])
		dst_image_path = os.path.join(dst_image_folder, file_name)
		if not os.path.exists(dst_image_path):
			tasks.append([url, dst_image_path])


	print('{} train images to be downloaded for {}'.format(len(train), category))
	print('{} validation images to be downloaded for {}'.format(len(validation), category))
	print('{} test images to be downloaded for {}'.format(len(test), category))


pool = ThreadPool(60)

STEP  = 100
start = 0
num   = len(tasks)

while start < num:
	end = min(num, start + STEP)

	# Add a block of tasks and wait till completion
	pool.map(download, tasks[start:end])
	pool.wait_completion()

	print ('{} Finished downloading {} / {} files'.format(strftime('%Y-%m-%d %H:%M:%S'), end, num))
	start = end
print("Done!")


print("{:<10} Start checking images".format('[INFO]'))
for image_path in [image_path_train, image_path_val, image_path_test]:
	for cate_folder in get_subfolder_paths(image_path):
		check_image(cate_folder)
print("{:<10} Finished checking images".format('[INFO]'))


for image_path in [image_path_train, image_path_val, image_path_test]:
	for cate_folder in get_subfolder_paths(image_path):
		print("{:<10} Images in {}: {}".format('[INFO]', cate_folder, len(get_sub_fnames(cate_folder))))

sys.exit(1)
