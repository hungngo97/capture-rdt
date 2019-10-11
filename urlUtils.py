from PIL import Image
import requests
from io import StringIO
import shutil
from utils import (
    resize_image,
    rotate_image,
    show_image,
    rotate_image1,
    resize_image_with_array,
    isLandscape,
    isVertical
)
import cv2 as cv


def readImageFromURL(url, isManualPhoto=False, output_path=''):
    response = requests.get(url)
    fileName = extractImageFileName(url)
    print('[Response] url', response, url, fileName)
    if response.status_code == 200:
        print('Writing..')
        with open(fileName, 'wb') as f:
            f.write(response.content)
    else:
        return 'NOT_FOUND'
    del response
    print('Finish writing...')
    print('isManualPhoto', isManualPhoto)
    if (isManualPhoto):
        print('Reading')
        img = cv.imread(fileName, cv.IMREAD_UNCHANGED)
        height, width = img.shape[:2]
        show_image(resize_image_with_array(
            img, gray=True, scale_percent=18))

        if isVertical(width, height):
            # rotate image
            print('[INFO] rotate manual photo')
            img = rotate_image1(img, 90)

        cv.imwrite(fileName, resize_image_with_array(
            img, gray=True, scale_percent=30))
        show_image(resize_image_with_array(
            rotate_image1(img, 90), gray=True, scale_percent=18))
    return fileName


def extractImageFileName(url):
    return url.split('/')[-1]
