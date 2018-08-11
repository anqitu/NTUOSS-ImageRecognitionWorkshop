# Reference: https://gist.github.com/kekeblom/204a609ee295c81c3cc202ecbe68752c
import os
import time
import argparse
import requests
import io
import numpy as np
from selenium import webdriver
from util import *

script_start_time = time.time()

def scroll_to_bottom(browser, num_requested):
    import math
    number_of_scrolls = math.ceil(num_requested / 400)

    for _ in range(number_of_scrolls):
    	for __ in range(10):
    		# multiple scrolls needed to show all 400 images
    		browser.execute_script("window.scrollBy(0, document.body.scrollHeight)")
    		time.sleep(0.2)
    	# to load next 400 images
    	time.sleep(2)
    	try:
            browser.find_element_by_xpath("//input[@value='Show more results']").click()
            print('Clicking "Show more results"')
    	except:
    		print("Cannot scrolling to bottom")
    		break

def fetch_image_urls(query, num_urls_to_extract):
    print('%0.2f min: Start fetching image urls for %s'%((time.time() - script_start_time)/60, query))
    site = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"
    browser = webdriver.Chrome(executable_path="/usr/local/bin/chromedriver")
    browser.get(site.format(q=query))
    scroll_to_bottom(browser, num_urls_to_extract)

    urls = []
    import json
    for x in browser.find_elements_by_xpath('//div[contains(@class,"rg_meta")]'):
        url = json.loads(x.get_attribute('innerHTML'))["ou"]
        urls.append(url)

    num_urls_extracted = len(urls)
    num_urls_to_return = min(num_urls_extracted, num_urls_to_extract)

    browser.quit()
    print('%0.2f min: Finish fetching image urls for %s'%((time.time() - script_start_time)/60, query))
    print("{:<10} Managed to get {} images. {} will be returned as requetsed".format('[INFO]', num_urls_extracted, num_urls_to_return))
    return urls[:num_urls_to_return]

def save_url_list_to_np(urls, path):
    import numpy as np
    x = np.array(urls)
    np.save(path, urls)


# Get project path
project_path = os.getcwd()
data_path = os.path.join(project_path, 'DataFile')
url_data_path = os.path.join(data_path, 'urls')

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description='Extract images urls from google image search')
    argument_parser.add_argument('--query', type=str, help='The query to download images from')
    argument_parser.add_argument('--count', default=100, type=int, help='How many images to fetch')
    argument_parser.add_argument('--label', type=str, help="The directory in which to store the images (images/<label>)", required=True)
    args = argument_parser.parse_args()

    ensure_directory(data_path)
    ensure_directory(url_data_path)

    urls = fetch_image_urls(args.query, args.count)
    url_path = os.path.join(url_data_path, args.label)
    save_url_list_to_np(urls, url_path)

    img = read_image_from_url(urls[0])
    plot_one_image(img, text = args.label)

    print("{:<10} Saved extracted {} urls of {} to {}".format('[CONGRATS]', len(urls), args.label, url_path))
    exit(1)
