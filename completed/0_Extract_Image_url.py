# Reference: https://gist.github.com/kekeblom/204a609ee295c81c3cc202ecbe68752c
import os
import time
import requests
import io
import numpy as np
from selenium import webdriver

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

def save_url_list_to_np(urls, label, path):
    import numpy as np
    x = np.array(urls)
    np.save(path, urls)

    print("{:<10} Saved extracted {} urls of {} to {}".format('[CONGRATS]', len(urls), label, url_path))


# Get project path
project_path = '/Users/anqitu/Workspaces/OSS/NTUOSS-ImageRecognitionWorkshop'
data_path = os.path.join(project_path, 'data')
url_data_path = os.path.join(data_path, 'urls')

if __name__ == '__main__':

    # use different query words to ensure sufficient images to feed the model.
    dog_query_words = ['dog', 'puppy', 'dog cute', 'dog big', 'dog small', 'dog yellow', 'dog white', 'dog black', 'dog brown', 'dog fluffy']
    cat_query_words = ['cat', 'cat lovely', 'cat small', 'cat big', 'cat white', 'cat black', 'cat yellow', 'cat fluffy', 'cat adorable', 'cat kitten']

    urls_list = []
    for query in dog_query_words:
        urls = fetch_image_urls(query, 1000)
        urls_list.extend(urls)
    urls = list(set(urls_list))
    len(urls)
    url_path = os.path.join(url_data_path, 'dog')
    save_url_list_to_np(urls, 'DOG', url_path)

    urls_list = []
    for query in cat_query_words:
        urls = fetch_image_urls(query, 1000)
        urls_list.extend(urls)
    urls = list(set(urls_list))
    len(urls)
    url_path = os.path.join(url_data_path, 'cat')
    save_url_list_to_np(urls, 'CAT', url_path)

    exit(1)
