import cv2 as cv
import numpy as np


def show_image(img, title="Example"):
    cv.imshow(title, img)
    cv.waitKey(0)


def rotate_image(img, degree, scale):
    h, w = img.shape
    center = (h/2, w/2)
    M = cv.getRotationMatrix2D(center, degree, scale)
    flipped = cv.warpAffine(img, M, (h, w))
    return flipped


def resize_image(src, gray=True, scale_percent=400):
    img = cv.imread(
        src, cv.IMREAD_GRAYSCALE if gray else cv.IMREAD_UNCHANGED)
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return resized
