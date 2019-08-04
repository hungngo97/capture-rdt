import cv2 as cv
import numpy as np
import os

FILES_TO_DELETE = [
    'cropResult.png',
    'result.png',
    'interpretResult.txt'
]


def show_image(img, title="Example"):
    cv.imshow(title, img)
    cv.waitKey(0)


def rotate_image(img, degree, scale):
    h, w = img.shape[:2]
    center = (h/2, w/2)
    M = cv.getRotationMatrix2D(center, degree, scale)
    flipped = cv.warpAffine(img, M, (h, w))
    return flipped


def rotate_image1(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (width/2, height/2)

    rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


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


def clear_files():
    for filename in FILES_TO_DELETE:
        os.remove(filename)


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
