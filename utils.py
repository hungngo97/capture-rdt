from __future__ import division
import cv2 as cv
import numpy as np
import os
import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array

FILES_TO_DELETE = [
    'cropResult.png',
    'result.png',
    'interpretResult.txt'
]


def calculateROCStats(truePositive, falsePositive, trueNegative, falseNegative):
    precision = calculatePrecisionScore(truePositive, falsePositive)
    recall = calculateRecallScore(truePositive, falseNegative)
    f1Score = calculateF1Score(precision, recall)
    falsePositiveRate = calculateFalsePositiveRate(
        falsePositive, trueNegative)
    truePositiveRate = calculateTruePositiveRate(
        truePositive, falseNegative)
    # Output
    print('F1 Score: ', f1Score)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('False Positive rate: ', falsePositiveRate)
    print('True Positive rate: ', truePositiveRate)
    return (f1Score, precision, recall, falsePositiveRate, truePositiveRate)


def calculatePrecisionScore(truePositive, falsePositive):
    if truePositive + falsePositive > 0:
        return truePositive / (truePositive + falsePositive)
    return 'N/A'


def calculateTruePositiveRate(truePositive, falseNegative):
    if truePositive + falseNegative > 0:
        return truePositive / (truePositive + falseNegative)
    return 'N/A'


def calculateFalsePositiveRate(falsePositive, trueNegative):
    if falsePositive + trueNegative > 0:
        return falsePositive / (falsePositive + trueNegative)
    return 'N/A'


def calculateRecallScore(truePositive, falseNegative):
    if truePositive + falseNegative > 0:
        return truePositive / (truePositive + falseNegative)
    return 'N/A'


def calculateF1Score(precision, recall):
    if precision == 'N/A' or recall == 'N/A':
        return 'N/A'
    if precision + recall > 0:
        return 2 * (precision * recall) / (precision + recall)
    return 'N/A'


def peakdet(v, delta, x=None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.

    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.

    """
    maxtab = []
    mintab = []

    if x is None:
        x = arange(len(v))

    v = asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if len(v) < 1:
        sys.exit('Array must have at least 1 element')

    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = v[0], v[0]
    mnpos, mxpos = None, None

    lookformax = True
    maxWidth = 0
    minWidth = 0

    for i in arange(1, len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
            maxWidth += 1
        if this < mn:
            mn = this
            mnpos = x[i]
            minWidth += 1

        if lookformax:
            if this < mx-delta:
                if mxpos != None:
                    """
                        TODO: width here might not be really accurate if there is a 
                        sharp increase from the left, and then a slower drop to the right,
                        the width will still be really small. We need a way to get the width from both 
                        sides of the peak
                    """
                    maxtab.append(
                        (mxpos, mx, findPeakWidth(mxpos, v, mx, findMax=True)))
                mn = this
                maxWidth = 0
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                if mnpos != NaN:
                    mintab.append(
                        (mnpos, mn, findPeakWidth(mnpos, v, mn, findMax=False)))
                mx = this
                minWidth = 0
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)


def findPeakWidth(idx, arr, val, findMax):
    width = 0
    if (findMax):
        # Find the furthest minimum left
        i = idx - 1
        # print(arr[i - 10: i + 10])
        # print(arr[i])
        while (i > 0 and arr[i] > arr[i - 1]):
            width += 1
            i -= 1
        # print('First', i, width)
        # Find the furthest minimum right
        i = idx
        while (i < len(arr) - 1 and arr[i] > arr[i + 1]):
            width += 1
            i += 1
        # print('Sec', i, width)
    else:
        # Find the furthest maximum left
        i = idx
        while (i > 0 and arr[i] < arr[i - 1]):
            width += 1
            i -= 1
        # Find the furthest maximum right
        i = idx
        while (i < len(arr) - 1 and arr[i] < arr[i + 1]):
            width += 1
            i += 1
    return width


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
