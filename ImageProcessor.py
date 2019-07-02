import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import enum
import time
from constants import (OVER_EXP_THRESHOLD, UNDER_EXP_THRESHOLD, OVER_EXP_WHITE_COUNT,
                       VIEW_FINDER_SCALE_H, VIEW_FINDER_SCALE_W, SHARPNESS_THRESHOLD,
                       CROP_RATIO, MIN_MATCH_COUNT)
from result import (ExposureResult, CaptureResult, InterpretationResult, SizeResult)
from utils import (show_image)

class ImageProcessor:
    def __init__(self, src):
        self.src = src
        self.img = cv.imread(src, cv.IMREAD_GRAYSCALE)
        height, width = self.img.shape
        self.height = height
        self.width = width
        self.featureDetector = cv.BRISK_create(45, 4, 1.0)
        self.matcher = cv.BFMatcher(cv.NORM_L1, crossCheck=False)
        # Load reference image for Quickvue flu test strip
        self.fluRefImg = cv.imread(
            'resources/quickvue_ref.jpg', cv.IMREAD_GRAYSCALE)
        # Load reference image for SD Bioline Malaria RDT
        self.malariaRefImg = cv.imread(
            '/resources/sd_bioline_malaria_ag_pf.jpg', cv.IMREAD_GRAYSCALE)
        print('[INIT] featureDetector, ', self.featureDetector)
        print('[INIT] img', self.fluRefImg.shape)
        refKeypoints, refDescriptors = self.featureDetector.detectAndCompute(
            self.fluRefImg, None)
        print('[INIT] Descriptors, ', refDescriptors)
        self.refKeypoints = refKeypoints
        self.refDescriptors = refDescriptors
        # Gaussian blur
        self.fluRefImg = cv.GaussianBlur(self.fluRefImg, (5, 5), 0)
        self.imageSharpness = self.calculateSharpness(self.fluRefImg)
        print('[INFO] flurefimg sharpness', self.imageSharpness)

        # Sift detector
        self.siftDetector = cv.xfeatures2d.SIFT_create()
        self.siftMatcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        refSiftKeypoints, refSiftDescriptors = self.siftDetector.detectAndCompute(
            self.fluRefImg, None)
        self.refSiftKeyPoints = refSiftKeypoints
        self.refSiftDescriptors = refSiftDescriptors
        print('[INIT] sift descriptors', self.refSiftDescriptors.shape)

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
        print('[INFO] Start Checksharpness')
        # Resized image
        (roiHeight, roiWidth) = img.shape
        resizeWidth = roiWidth * self.width / roiWidth
        resizeHeight = roiHeight * self.height / roiHeight
        dim = (resizeWidth, resizeHeight)
        resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
        print('[INFO] resized ROI shape: ', resized.shape)
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

    def detectRDT(self, img):
        print('[INFO] start detectRDT')
        startTime = time.time()

        height, width = img.shape
        mask = np.zeros(img.shape, np.uint8)
        p1 = (0, int(height * (1 - VIEW_FINDER_SCALE_W / CROP_RATIO) / 2))
        p2 = (int(width - p1[0]), int(height - p1[1]))
        # mask[p1[0]: p2[0], p1[1]: p2[1]] = np.full(
        #     (p2[0] - p1[0], p2[1] - p1[1]), 255
        # )
        mask[p1[0]: p2[0], p1[1]: p2[1]] = 255

        keypoints, descriptors = self.siftDetector.detectAndCompute(img, mask)
        print('[INFO] detect/compute time: ', time.time() - startTime)
        print('[INFO] descriptors')
        print(descriptors)
        # if (descriptors == None or all(descriptors)):
        #     print('[WARNING] No Features on input')
        #     return None

        # Matching
        matches = self.matcher.match(self.refSiftDescriptors, descriptors)
        print('[INFO] Finish matching')
        matches = sorted(matches, key=lambda x: x.distance)

        matchingImage = cv.drawMatches(self.fluRefImg, self.refSiftKeyPoints, img,
                                       keypoints, matches[:50], None, flags=2)
        # plt.imshow(matchingImage)
        # plt.title('SIFT Brute Force matching')
        # plt.show()

        sum = 0
        distance = 0
        count = 0

        # store all the good matches as per Lowe's ratio test.
        good = []
        img2 = None
        dst = None
        good = matches[:20]
        # for m in matches:
        #     if m.distance < 0.7:
        #         good.append(m)
        if len(good)> MIN_MATCH_COUNT:
            src_pts = np.float32([ self.refSiftKeyPoints[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ keypoints[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
            print('[INFO] Finish finding Homography')
            print('[INFO] M Matrix', M)
            print('[INFO] mask', mask)
            matchesMask = mask.ravel().tolist()
            h,w = self.fluRefImg.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts,M)
            print('[INFO] dst transformation pts', dst)
            img2 = cv.polylines(img,[np.int32(dst)],True,(255,0,0))
            print('[INFO] finish perspective transform')
            # plt.imshow(img2)
            # plt.show()
        else:
            print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            matchesMask = None
            return None

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
        img3 = cv.drawMatches(self.fluRefImg,self.refSiftKeyPoints,img2,keypoints,good,None,**draw_params)
        plt.imshow(img3, 'gray'),plt.show()
        return dst 


    def getBoundarySize(boundary):
        xMax = -1
        xMin = -1 
        yMin = -1 
        yMax = -1
        for pt in boundary:
            point = pt[0]
            x, y = point[0], point[1]
            xMax = max(xMax, x)
            xMin = min(xMin, x)
            yMin = min(yMin, y)
            yMax = max(yMax, y)
        return (xMax - xMin, yMax - yMin)

    def checkIfCentered(self, boundary, imgShape):
        print('[INFO] checkIfCentered')

    def checkSize(self, boundary, imgShape):
        print('[INFO] checkSize')

    def checkOrientation(self, boundary):
        print('[INFO] checkIfCentered')

    def measureOrientation(self, boundary):
        print('[INFO] checkIfCentered')

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

        if (exposureResult == ExposureResult.NORMAL and isSharp):
            boundary = detectRDT(img)
            isCentered = False
            sizeResult = SizeResult.INVALID
            isRightOrientation = False
            angle = 0.0
            w, h = getBoundarySize(boundary)

            if (w > 0 and h > 0):
                isCentered = checkIfCentered(boundary, img.shape)
                sizeResult = checkSize(boundary, img.shape)
                isRightOrientation = checkOrientation(boundary)
                angle = measureOrientation(boundary)
            passed = sizeResult == SizeResult.RIGHT_SIZE and isCentered and isRightOrientation
            return CaptureResult()
        else:
            return CaptureResult()
