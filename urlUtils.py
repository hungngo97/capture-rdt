from PIL import Image
import requests
from io import StringIO
import shutil
from utils import (
    resize_image
)
import cv2 as cv


def readImageFromURL(url):
    response = requests.get(url)
    fileName = extractImageFileName(url)
    print('[Response] url', response, url, fileName)
    if response.status_code == 200:
        with open(fileName, 'wb') as f:
            f.write(response.content)
    else:
        return 'NOT_FOUND'
    del response
    resize_image(fileName, scale_percent=50)
    return fileName


def extractImageFileName(url):
    return url.split('/')[-1]
