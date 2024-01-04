# Colorisation d'images via le machine learning

![colorized image](images_folder/image%20bank/results/1.jpg)


*Colorized image*


## Description of the puzzle app

Explain the puzzle (readme preparing merging)

## How to run the app
First, build the docker image by running the following command in a terminal :

```bash
docker build . -t puzzle_image
```

Then, run the container:

```bash
docker run -p 8000:8000 puzzle_image
```

The app runs on `http://0.0.0.0:8000/`. If this url does not work, try `http://127.0.0.1:8000/` or `http://localhost:8000/`.

## Scripts behind the app : organization of the repository
To develop the app, we followed several steps.
* **Scrap images** : to obtain enough images to train the model, we scrapped images on Google Image. The scripts for this part are available in the folder `scraping_images`. `scraping.py` does the scraping, and `color_to_bw.py` resizes the images and converts them to black and white. The commands to use to run them are specified at the top of each script. The images are stored in the subfolder `images`, and are in separate zip files in order to stay under the maximum file size in GitHub.
* **Train a colorization model** : we then trained a Convolutional Neural Network on the images in order to set up a model able to colorize greyscale images. The training is done in the folder `colorization_model`. The script `colorization_training.py` trains the CNN and saves the optimal parameters in `model1.pth`.
* ...

## Notes on running the scripts
The scripts relevant to the scraping and the training of the images don't have to run in order to run the app. Indeed, the model is already trained and does not need to be trained again. However, if you want to run these scripts, there are a few actions to take first:
* **Scraping** : a chrome webdriver is needed to run the script. To download it, see https://chromedriver.chromium.org/downloads.
* **Model training** : before running the script to train the model, unzip all color images (that can be found in `scraping_images/images`) into the subfolder `images_folder/color/color_images`.
