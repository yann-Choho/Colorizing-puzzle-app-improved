from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time
import requests
import io
from PIL import Image

#Webdriver path
webdriver_path = "./chromedriver-win64/chromedriver.exe"




#Get images urls for the search word "key"
def get_images_urls(driver, key, max_images=200, delay=10, verbose=True):

    
    #Find the search bar, type the key, and enter
    if len(driver.find_elements_by_id('REsRA')) != 0: #If the driver is on a google image page
        search = driver.find_element_by_id('REsRA')
        search.clear()
        search.send_keys(key)
        search.send_keys(Keys.RETURN)
    else: #If we are in the homepage of google image
        #Find the search bar
        search = driver.find_element(By.ID,'APjFqb')
        #Type in the search bar
        search.send_keys(key)
        #Click the "search" button
        driver.find_element(By.CSS_SELECTOR,"[aria-label='Recherche Google']").click()
        
    #Get urls
    urls = set()
    thumbnails = WebDriverWait(driver, delay).until(EC.presence_of_all_elements_located((By.CLASS_NAME, "Q4LuWd")))
    if verbose:
        print(f'Number of thumbnails for key {key} : {len(thumbnails)}')


    for tn in thumbnails:
        if len(urls) >= max_images:
            break
        try:
            tn.click()
            tn_images = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.ID, 'islsp'))) #Go into the frame of the image we clicked on
            time.sleep(0.7)
            images = tn_images.find_elements_by_xpath(".//*[@class='rg_i Q4LuWd' or @class='sFlh5c pT0Scc iPVvYb']")
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
    try:
        #Get the image url and convert to image
        img_content = requests.get(url).content
        img_file = io.BytesIO(img_content)
        image = Image.open(img_file)

        with open(file_path + file_name + '.png', 'wb') as file:
            image.save(file, file_type)

        if verbose == True:
            print(f'{file_name} downloaded successfully.')
    except Exception as e:
        print(f'Unsuccessful download : \n {str(e)}')



def get_images(keys, webdriver_path, download_path, max_images_per_key=200, delay=10, file_type='PNG', verbose=True):
    #Find chrome webdriver
    driver = webdriver.Chrome(webdriver_path)

    #Go to google images
    driver.get("https://images.google.fr/")

    #Reject cookies
    driver.find_element(By.ID,"W0wltc").click()

    for key in keys:
        img_nb = 1
        urls = get_images_urls(driver=driver, key=key, max_images=max_images_per_key, delay=delay, verbose=verbose)
        for url in urls:
            download_image(url=url, file_path=download_path, file_name=key + str(img_nb), file_type=file_type, verbose=verbose)
            img_nb += 1
    
    driver.close()


keys = ['horse', 'car', 'forest']
get_images(keys, webdriver_path, './images/color_images/', max_images_per_key=50)