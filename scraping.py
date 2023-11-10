from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time
import requests
import io
from PIL import Image



def get_images_urls(driver, key, max_images=200, delay=10, verbose=True):
    """Get images urls for the search word "key".

    Arguments:
        driver (webdriver) : Google Chrome webdriver, on a google image page.
        key (string) : search word (e.g. "horse") to look for images.
        max_images (int) : maximum number of images urls we want to have for the search word "key".
        delay (int) : maximum time (in seconds) we are willing to wait for a page to load.
        verbose (boolean) : if True, print strings describing the operations of the code.

    Returns:
        urls (string set) : set of at most max_images of images urls.
    """

    
    #Find the search bar, type the key, and enter
    if len(driver.find_elements_by_id('REsRA')) != 0: #If the driver is on a google image page that is not the homepage
        #Find the search bar, clear it, type the keyword "key" and hit Enter
        search = driver.find_element_by_id('REsRA')
        search.clear()
        search.send_keys(key)
        search.send_keys(Keys.RETURN)
    else: #If the driver is on the homepage of google image
        #Find the search bar, type the keyword "key" and click the "search" button
        search = driver.find_element(By.ID,'APjFqb')
        search.send_keys(key)
        driver.find_element(By.CSS_SELECTOR,"[aria-label='Recherche Google']").click()
        
    urls = set()

    #thumbnails : list of web elements corresponding to the thumbnails of each image on the page
    thumbnails = WebDriverWait(driver, delay).until(EC.presence_of_all_elements_located((By.CLASS_NAME, "Q4LuWd")))
    if verbose:
        print(f'Number of thumbnails for key {key} : {len(thumbnails)}')


    for tn in thumbnails:
        if len(urls) >= max_images:
            break
        try:
            tn.click()
            #Go into the frame of the image we clicked on
            tn_images = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.ID, 'islsp')))
            time.sleep(0.7)
            #images : web element corresponding to the images in the frame (which have an attribute 'src')
            images = tn_images.find_elements_by_xpath(".//*[@class='rg_i Q4LuWd' or @class='sFlh5c pT0Scc iPVvYb']")
            #Add each image url to urls (except if there are more than max_images in urls)
            for image in images:
                url = image.get_attribute('src')
                if len(urls) >= max_images:
                    break
                if (url != None) and (url not in urls) and ('http' in url):
                    urls.add(url)
        except:
            if verbose:
                print("No images for this thumbnail")
            continue
    return urls


def download_image(url, file_path, file_name, file_type='PNG', verbose=True):
    """Download an image given its url.

    Arguments:
        url (string) : url of the image.
        file_path (string) : path of the folder to store the file in.
        file_name (string) : name of the file.
        file_type (string) : type of the image ('PNG', 'JPEG',...).
        verbose (boolean) : if True, print strings describing the operations of the code.
    """
    try:
        #Get the image url and convert to image
        img_content = requests.get(url).content
        img_file = io.BytesIO(img_content)
        image = Image.open(img_file)

        #Save the image
        with open(file_path + file_name + '.' + file_type.lower(), 'wb') as file:
            image.save(file, file_type)

        if verbose:
            print(f'{file_name} downloaded successfully.')

    except Exception as e:
        print(f'Unsuccessful download of {file_name} : \n {str(e)}')
        pass



def get_images(keys, webdriver_path, download_path, max_images_per_key=200, delay=10, file_type='PNG', verbose=True):
    """Find and download images given a list of key words.

    Arguments:
        keys (string list) : list of key words to look for.
        webdriver_path (string) : path of a Chrome webdriver.
        download_path (string) : path of the folder to store the images in.
        max_images_per_key (int) : maximum number of images we want to have for each key word.
        delay (int) : maximum time (in seconds) we are willing to wait for a page to load.
        file_type (string) : type of the images ('PNG', 'JPEG',...).
        verbose (boolean) : if True, print strings describing the operations of the code.
    """
    #Find chrome webdriver
    driver = webdriver.Chrome(webdriver_path)

    #Go to google images
    driver.get("https://images.google.fr/")

    #Reject cookies if necessary
    try:
        driver.find_element(By.ID,"W0wltc").click()
    except:
        pass

    #Download at most max_images_per_key for each key
    for key in keys:
        img_nb = 1
        urls = get_images_urls(driver=driver, key=key, max_images=max_images_per_key, delay=delay, verbose=verbose)
        for url in urls:
            download_image(url=url, file_path=download_path, file_name=key + str(img_nb), file_type=file_type, verbose=verbose)
            img_nb += 1
    
    driver.close()


#Launch code
webdriver_path = "./chromedriver-win64/chromedriver.exe"
keys = ['chocolate', 'ice cream']
download_path = './images/color_images/'
get_images(keys, webdriver_path, download_path, max_images_per_key=20)