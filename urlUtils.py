from PIL import Image
import requests
from io import StringIO
import shutil
from utils import (
    resize_image,
    rotate_image,
    show_image,
    rotate_image1
)
import cv2 as cv


def readImageFromURL(url, isManualPhoto=False, output_path=''):
    response = requests.get(url)
    fileName = extractImageFileName(url)
    print('[Response] url', response, url, fileName)
    if response.status_code == 200:
        with open(fileName, 'wb') as f:
            f.write(response.content)
    else:
        return 'NOT_FOUND'
    del response
    if (isManualPhoto):
        img = cv.imread(fileName, cv.IMREAD_UNCHANGED)
        print('[INFO] rotate manual photo')
        cv.imwrite(fileName, rotate_image1(img, 90))
    return fileName


def extractImageFileName(url):
    return url.split('/')[-1]
