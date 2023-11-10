import os
from PIL import Image


def convert_color_to_black_and_white(path_color, path_bw, file_type='PNG'):
    """Convert color images to black and white images, and stores them in path_bw.

    Arguments:
        path_color (string) : path of the folder in which the color images are.
        path_bw (string) : path of the folder in which to store the black and white images.
        file_type (string) : type of the images ('PNG', 'JPEG',...).
    """

    for img_name in os.listdir(path_color):
        color_img = Image.open(path_color + img_name)
        bw_img = color_img.convert('L')

        with open(path_bw + img_name, 'wb') as file:
            bw_img.save(file, file_type)


#Convert the color images downloaded in scraping.py
convert_color_to_black_and_white(path_color="./images/color_images/",
                                 path_bw="./images/bw_images/")