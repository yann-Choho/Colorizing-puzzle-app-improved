import os
from PIL import Image


def convert_color_to_black_and_white(path_color, path_bw, file_type='PNG'):
    for img_name in os.listdir(path_color):
        color_img = Image.open(path_color + img_name)
        bw_img = color_img.convert('L')

        with open(path_bw + img_name, 'wb') as file:
            bw_img.save(file, file_type)



convert_color_to_black_and_white(path_color="./images/color_images/",
                                 path_bw="./images/bw_images/")