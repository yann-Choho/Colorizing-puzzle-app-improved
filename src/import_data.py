# -*- coding: utf-8 -*-

### === Script for data import === ###


# Requirements
import os
import requests
import yaml



def import_yaml_config(filename: str = "null.yaml") -> dict:
    """Function to load the configuration data

    Args:
        filename (str): name of the configuration file

    Returns:
        dict_config: the configuration informations as a dictionnary
    """
    dict_config = {}
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as stream:
            dict_config = yaml.safe_load(stream)
    return dict_config



def download_file(url: str, local_dir: str):
    """Function to upload a file from a download URL to this project 

    Args:
        url (str): the download URL of a file
        local_dir (str): the adequate data folder in the project 

    Returns:
        Nothing (it saves the data in the right folder)
    """
    filename = os.path.basename(url)
    local_path = os.path.join(local_dir, filename)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)



def download_files(url_base: str, local_dir: str, num_files: int):  
    """Function to loop over the files to import 

    Args:
        url_base (str): the download URL of the distant database
        local_dir (str): the adequate data folder in the project 
        num_files (int): the number of files to loop over

    Returns:
        Nothing (it calls the download_file function)
    """
    os.makedirs(local_dir, exist_ok=True)

    for i in range(1, num_files + 1):
        file_url = f"{url_base}/_{i}.jpg" 
        download_file(file_url, local_dir)