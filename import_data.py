import os
import shutil
import requests
import yaml
from urllib.parse import urlparse



def import_yaml_config(filename: str = "null.yaml") -> dict:
    dict_config = {}
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as stream:
            dict_config = yaml.safe_load(stream)
    return dict_config



def download_file(url, local_dir):
    filename = os.path.basename(url)
    local_path = os.path.join(local_dir, filename)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)



def download_files(url_base, local_dir, num_files):
    os.makedirs(local_dir, exist_ok=True)

    for i in range(1, num_files + 1):
        file_url = f"{url_base}/_{i}.jpg" 
        download_file(file_url, local_dir)