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


def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center = rect[0]
    size = rect[1]
    angle = rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]
    print("width: {}, height: {}".format(width, height))

    M = cv.getRotationMatrix2D(center, angle, 1)
    img_rot = cv.warpAffine(img, M, (width, height))

    img_crop = cv.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot


class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Rect():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
