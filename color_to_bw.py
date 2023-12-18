#How to run : python color_to_bw.py path_original path_color path_bw
#Ex : python color_to_bw.py ./images/original_images/ ./images/color_images/ ./images/bw_images/

import sys
import os
from shutil import make_archive
from PIL import Image

def resize(img: Image.Image, width: int = 64, height: int = 64) -> Image.Image:
    """Resize an image to a specified width and height, crop image if necessary to maintain aspect ratio.
    
    Arguments:
        img (Image.Image) : image to resize.
        width (int) : width of the resized image (in pixels).
        height (int) : height of the resized image (in pixels).

    Returns:
        resized_img (Image.Image) : resized image.
    """

    ratio = width/height
    w, h = img.size
    #If the width is to big, crop the right of the image (the dimensions of the image must be proportional to the specified width and height, ie w/h = width/height).
    if w/h > ratio:
        cropped_img = img.crop((0, 0, h*ratio, h))
    #If the height is too big, crop the top of the image.
    else:
        cropped_img = img.crop((0, 0, w, w/ratio))
    #Resize the cropped image to have the specified width and height
    resized_img = cropped_img.resize((width, height))

    return resized_img


def resize_images(path_original: str, path_resized: str, width: int = 64, height: int = 64, file_type: str = 'PNG') -> None:
    """Resize color images to a specified width and height, save resized images in path_resized.
    
    Arguments:
        path_original (str) : path of the folder in which the scraped images are.
        path_resized (str) : path of the folder in which to store the resized color images.
        width (int) : width of the resized image (in pixels).
        height (int) : height of the resized image (in pixels).
        file_type (str) : type of the images ('PNG', 'JPEG',...).
    """
    if not os.path.exists(path_resized):
        os.makedirs(path_resized)

    for img_name in os.listdir(path_original):
        img = Image.open(path_original + img_name)
        resized_img = resize(img, width, height)
    
        #Save resized image
        with open(path_resized + img_name, 'wb') as file:
            resized_img.save(file, file_type)
    
    #Zip the folder
    make_archive(path_resized[:-1], format='zip', root_dir=path_resized)


def convert_color_to_black_and_white(path_color: str, path_bw: str, file_type: str = 'PNG') -> None:
    """Convert color images to black and white images, and stores them in path_bw.

    Arguments:
        path_color (str) : path of the folder in which the color images are.
        path_bw (str) : path of the folder in which to store the black and white images.
        file_type (str) : type of the images ('PNG', 'JPEG',...).
    """

    if not os.path.exists(path_bw):
        os.makedirs(path_bw)

    for img_name in os.listdir(path_color):
        color_img = Image.open(path_color + img_name)
        bw_img = color_img.convert('L')

        with open(path_bw + img_name, 'wb') as file:
            bw_img.save(file, file_type)
    
    #Zip the folder
    make_archive(path_bw[:-1], format='zip', root_dir=path_bw)


if __name__ == "__main__":
    path_original = sys.argv[1]
    path_color = sys.argv[2]
    path_bw = sys.argv[3]

    resize_images(path_original=path_original,
                path_resized=path_color,
                width=64,
                height=64)

    convert_color_to_black_and_white(path_color=path_color,
                                    path_bw=path_bw)