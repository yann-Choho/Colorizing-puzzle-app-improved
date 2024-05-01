import pytest
from PIL import Image
from color_to_bw import resize

def test_resize():
    """ Test the resizing of an example images """
    img = Image.open('./scraping_images/tests/forest308.png')
    assert resize(img, 30, 200).size == (30, 200)
    assert resize(img, 64, 32).size == (64, 32)