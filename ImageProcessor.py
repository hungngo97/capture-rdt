import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import enum
from constants import (OVER_EXP_THRESHOLD, UNDER_EXP_THRESHOLD, OVER_EXP_WHITE_COUNT,
                       VIEW_FINDER_SCALE_H, VIEW_FINDER_SCALE_W, SHARPNESS_THRESHOLD)


class ExposureResult(enum.Enum):
    UNDER_EXPOSED = 0
    NORMAL = 1
    OVER_EXPOSED = 2


class ImageProcessor:
    def __init__(self, src):
        self.src = src
        self.img = cv.imread(src, cv.IMREAD_GRAYSCALE)
        height, width = self.img.shape
        self.height = height
        self.width = width
        self.featureDetector = cv.BRISK_create(45, 4, 1.0)
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        # Load reference image for Quickvue flu test strip
        self.fluRefImg = cv.imread(
            'resources/quickvue_ref.jpg', cv.IMREAD_GRAYSCALE)
        # Load reference image for SD Bioline Malaria RDT
        self.malariaRefImg = cv.imread(
            '/resources/sd_bioline_malaria_ag_pf.jpg', cv.IMREAD_GRAYSCALE)
        keypoints, descriptors = self.featureDetector.detectAndCompute(
            self.fluRefImg, None)
        # Gaussian blur
        self.fluRefImg = cv.GaussianBlur(self.fluRefImg, (5, 5), 0)
        self.imageSharpness = self.calculateSharpness(self.fluRefImg)
        print('[INFO] flurefimg sharpness', self.imageSharpness)

        # Sift detector
        self.siftDetector = cv.xfeatures2d.SIFT_create()
        self.siftMatcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        siftKeypoints, siftDescription = self.siftDetector.detectAndCompute(
            self.fluRefImg, None)

    def checkBrightness(self, img):
        histograms = self.calculateBrightness(img)
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

    def calculateBrightness(self, src):
        histSizeNum = 256
        histSize = 256
        histRange = (0, 256)  # the upper boundary is exclusive
        accumulate = False
        height, width = src.shape

        hist = cv.calcHist([self.img], [0], None, [256], [0, 256])
        hist_h = 400
        cv.normalize(hist, hist, alpha=height/2, beta=0,
                     norm_type=cv.NORM_INF)
        # print(hist.shape)
        # plt.plot(hist)
        # plt.title('Histogram of grey scale pixels')
        # plt.xlabel('Pixel Value')
        # plt.ylabel('Amount')
        # plt.show()
        return hist

    def getViewfinderRect(self, img):
        height, width = img.shape
        p1 = (int(width * (1 - VIEW_FINDER_SCALE_H)/2),
              int(height * (1 - VIEW_FINDER_SCALE_W) / 2))
        p2 = (int(width - p1[0]), int(height - p1[1]))
        return (p1, p2)

    def checkSharpness(self, img):
        """
            resize(inputMat, resized, new Size(inputMat.size().width*mRefImg.size().width/inputMat.size().width, inputMat.size().height*mRefImg.size().width/inputMat.size().width));

            double sharpness = calculateSharpness(resized);
            //Log.d(TAG, String.format("inputMat sharpness: %.2f, %.2f",calculateSharpness(resized), calculateSharpness(inputMat)));
            Log.d(TAG, String.format("inputMat sharpness: %.2f",calculateSharpness(resized)));

            boolean isSharp = sharpness > (refImgSharpness * (1-SHARPNESS_THRESHOLD));
            Log.d(TAG, "Sharpness: "+sharpness);

            resized.release();

            return isSharp;
        """
        print('[INFO] Start Checksharpness')
        # Resized image
        (roiHeight, roiWidth) = img.shape
        resizeWidth = roiWidth * self.width / roiWidth
        resizeHeight = roiHeight * self.height / roiHeight
        dim = (resizeWidth, resizeHeight)
        resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
        print('[INFO] resized ROI shape: ', resized.shape)
        cv.imshow('ROI', img)
        cv.waitKey(0)
        sharpness = self.calculateSharpness(resized)
        isSharp = sharpness > (float('-inf') * (1 - SHARPNESS_THRESHOLD))
        print('[INFO] isSharp: ', isSharp)
        return isSharp

    def calculateSharpness(self, img):
        print('[INFO] start calculateSharpness')
        # [laplacian]
        # Apply Laplace function
        laplacianResult = cv.Laplacian(img, cv.CV_64F)
        print('[INFO] laplacianResult ', laplacianResult.shape)
        mean, stddev = cv.meanStdDev(laplacianResult)
        print('[INFO] stddev ', stddev, stddev.shape)
        sharpness = stddev[0][0] ** 2
        print('[INFO] sharpness ', sharpness)
        return sharpness

    def captureRDT(self, src):
        img = cv.imread(src, cv.IMREAD_GRAYSCALE)
        # Check brightness
        exposureResult = (self.checkBrightness(img))
        # check sharpness (refactored)
        (x1, y1), (x2, y2) = self.getViewfinderRect(img)

        print(x1, y1, x2, y2)
        roi = img[x1:x2, y1:y2]
        print(roi.shape)
        isSharp = self.checkSharpness(roi)
