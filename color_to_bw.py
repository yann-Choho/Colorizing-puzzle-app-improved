import sys
import os
from PIL import Image

#How to run : python color_to_bw.py path_original path_color path_bw
#Ex : python color_to_bw.py ./images/original_images/ ./images/color_images/ ./images/bw_images/
path_original = sys.argv[1]
path_color = sys.argv[2]
path_bw = sys.argv[3]

def resize_images(path_original, path_resized, width=300, height=200, file_type='PNG'):
    """Resize color images to a specified width and height, crop image if necessary to maintain aspect ratio. Save resized images in path_resized.
    
    Arguments:
        path_original (string) : path of the folder in which the scraped images are.
        path_resized (string) : path of the folder in which to store the resized color images.
        width (int) : width of the resized image (in pixels).
        height (int) : height of the resized image (in pixels).
        file_type (string) : type of the images ('PNG', 'JPEG',...).
    """
    ratio = width/height
    for img_name in os.listdir(path_original):
        img = Image.open(path_original + img_name)
        w, h = img.size
        #If the width is to big, crop the right of the image (the dimensions of the image must be proportional to the specified width and height, ie w/h = width/height).
        if w/h > ratio:
            cropped_img = img.crop((0, 0, h*ratio, h))
        #If the height is too big, crop the top of the image.
        else:
            cropped_img = img.crop((0, 0, w, w/ratio))
        #Resize the cropped image to have the specified width and height
        resized_img = cropped_img.resize((width, height))
    
        #Save resized image
        with open(path_resized + img_name, 'wb') as file:
            resized_img.save(file, file_type)



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


resize_images(path_original=path_original,
              path_resized=path_color,
              width=300,
              height=200)

convert_color_to_black_and_white(path_color=path_color,
                                 path_bw=path_bw)