import cv2
import numpy as np
import matplotlib.pyplot as plt
import enum
from constants import OVER_EXP_THRESHOLD, UNDER_EXP_THRESHOLD, OVER_EXP_WHITE_COUNT


class ExposureResult(enum.Enum):
    UNDER_EXPOSED = 0
    NORMAL = 1
    OVER_EXPOSED = 2


def checkBrightness(img):
    histograms = calculateBrightness(img)
    maxWhite = 0
    whiteCount = 0
    length = np.shape(histograms)[0]
    whiteCount = histograms[length - 1]
    for i in range(length):
        if histograms[i] > 0:
            maxWhite = i

    if maxWhite >= OVER_EXP_THRESHOLD and whiteCount > OVER_EXP_WHITE_COUNT:
        return ExposureResult.OVER_EXPOSED
    elif maxWhite < UNDER_EXP_THRESHOLD:
        return ExposureResult.UNDER_EXPOSED
    return ExposureResult.NORMAL


def calculateBrightness(src):
    histSizeNum = 256
    histSize = 256
    histRange = (0, 256)  # the upper boundary is exclusive
    accumulate = False
    height, width = src.shape

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_h = 400
    cv2.normalize(hist, hist, alpha=height/2, beta=0,
                  norm_type=cv2.NORM_INF)
    # print(hist.shape)
    # plt.plot(hist)
    # plt.title('Histogram of grey scale pixels')
    # plt.xlabel('Pixel Value')
    # plt.ylabel('Amount')
    # plt.show()
    return hist


INPUT_IMAGE = 'input/testimg1.jpg'

img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
# print(img.shape)
# cv2.imshow('Img', img)
# cv2.waitKey(0)

# Check brightness
exposureResult = (checkBrightness(img))
