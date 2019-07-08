import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import enum
import time
from constants import (OVER_EXP_THRESHOLD, UNDER_EXP_THRESHOLD, OVER_EXP_WHITE_COUNT,
                       VIEW_FINDER_SCALE_H, VIEW_FINDER_SCALE_W, SHARPNESS_THRESHOLD,
                       CROP_RATIO, MIN_MATCH_COUNT, POSITION_THRESHOLD, ANGLE_THRESHOLD,
                       CONTROL_LINE_POSITION, TEST_A_LINE_POSITION, TEST_B_LINE_POSITION,
                       CONTROL_LINE_COLOR_UPPER, CONTROL_LINE_COLOR_LOWER, CONTROL_LINE_POSITION_MIN,
                       CONTROL_LINE_POSITION_MAX, CONTROL_LINE_MIN_HEIGHT, CONTROL_LINE_MIN_WIDTH,
                       CONTROL_LINE_MAX_WIDTH)
from result import (ExposureResult, CaptureResult, InterpretationResult, SizeResult)
from utils import (show_image, resize_image, Point, Rect)

class ImageProcessor:
    def __init__(self, src):
        self.src = src
        self.img = cv.imread(src, cv.IMREAD_GRAYSCALE)
        width, height = self.img.shape
        self.height = height
        self.width = width
        self.featureDetector = cv.BRISK_create(45, 4, 1)
        self.matcher = cv.BFMatcher(cv.NORM_L1, crossCheck=False)
        # Load reference image for Quickvue flu test strip
        self.fluRefImg = resize_image(
            'resources/quickvue_ref_v4_1.jpg', gray=True, scale_percent=400)
        # show_image(self.fluRefImg)
        # print(self.fluRefImg)
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
        width, height = src.shape

        hist = cv.calcHist([self.img], [0], None, [256], [0, 256])
        hist_h = 400
        cv.normalize(hist, hist, alpha=height/2, beta=0,
                     norm_type=cv.NORM_INF)
        print(hist.shape)
        # plt.plot(hist)
        # plt.title('Histogram of grey scale pixels')
        # plt.xlabel('Pixel Value')
        # plt.ylabel('Amount')
        # plt.show()
        return hist

    def getViewfinderRect(self, img):
        width, height = img.shape
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

    def detectRDT(self, img, cnt=5):
        print('[INFO] start detectRDT')
        startTime = time.time()
        # show_image(img)
        width, height = img.shape
        p1 = (0, int(height * (1 - VIEW_FINDER_SCALE_W / CROP_RATIO) / 2))
        p2 = (int(width - p1[0]), int(height - p1[1]))
        keypoints, descriptors = self.siftDetector.detectAndCompute(img, None)
        # keypoints, descriptors = self.siftDetector.detectAndCompute(img, mask)
        print('[INFO] detect/compute time: ', time.time() - startTime)
        print('[INFO] descriptors')
        print(descriptors)
        # TODO: Find condition for this
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
        good = matches[:50]
        print('[INFO] matches')
        # for m in matches:
        #     if m.distance < 0.7:
        #         good.append(m)
        if len(good)> MIN_MATCH_COUNT:
            src_pts = np.float32([ self.refSiftKeyPoints[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ keypoints[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            print('src_pts', src_pts)
            print('dst_pst', dst_pts)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, cnt)
            print('[INFO] Finish finding Homography')
            print('[INFO] M Matrix', M)
            # print('[INFO] mask', mask)
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

        draw_params = dict(matchColor = (255,0,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
        img3 = cv.drawMatches(self.fluRefImg,self.refSiftKeyPoints,img2,keypoints,good,None,**draw_params)
        # plt.imshow(img3, 'gray'),plt.show()
        return dst 


    def getBoundarySize(self, boundary):
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

    def checkIfCentered(self, boundary, imgShape, img):
        print('[INFO] checkIfCentered')
        center = self.measureCenter(boundary, img)
        print('[INFO] rotated rect center', center)
        w, h  = imgShape
        trueCenter = (w/2 , h/2)
        isCentered = center[0] < trueCenter[0] + (w * POSITION_THRESHOLD) and center[0] > trueCenter[0]-(w *POSITION_THRESHOLD) and center[1] < trueCenter[1]+(h *POSITION_THRESHOLD) and center[1] > trueCenter[1]-(h*POSITION_THRESHOLD)
        print('[INFO] isCentered:', isCentered)
        # TODO: draw image to show how to get isCentered
        return isCentered
        

    def measureCenter(self, boundary, img):
        rect = cv.minAreaRect(boundary)
        print('[INFO] rotated rect', rect)
        # box = cv.boxPoints(rect)
        # box = np.int0(box)
        # show_image(img)
        # cv.drawContours(img,[box],0,(0,0,255),2)
        # show_image(img)

        # TODO: Maybe we can visualize how this rectangle is compared to the original rect
        return rect[0]

    def checkSize(self, boundary, imgShape):
        print('[INFO] checkSize')
        width, height = imgShape
        largestDimension = self.measureSize(boundary)
        print('[INFO] height', height)
        # TODO: not sure if we even need the view finder scale??
        isRightSize = largestDimension < width * VIEW_FINDER_SCALE_H + 100 and largestDimension > width * VIEW_FINDER_SCALE_H - 100
        
        sizeResult = SizeResult.INVALID
        if isRightSize:
            sizeResult = SizeResult.RIGHT_SIZE
        elif largestDimension > height * VIEW_FINDER_SCALE_H + 100:
                sizeResult = SizeResult.LARGE
        elif largestDimension < height * VIEW_FINDER_SCALE_H - 100:
            sizeResult = SizeResult.SMALL
        else:
            sizeResult = SizeResult.INVALID
        print('[INFO] sizeResult', sizeResult)
        return sizeResult

    
    def measureSize(self, boundary):
        (x, y), (width, height), angle = cv.minAreaRect(boundary)
        isUpright = height > width
        angle = 0
        h = 0
        if (isUpright):
            angle = 90 - abs(angle)
        else:
            angle = abs(angle)
        return  height if isUpright else width

    def checkOrientation(self, boundary):
        print('[INFO] checkIfCentered')
        angle = self.measureOrientation(boundary)
        print('[INFO] measured angle', angle)
        isOriented = abs(angle) < ANGLE_THRESHOLD
        print('[INFO] isOriented', isOriented)
        return isOriented

    def measureOrientation(self, boundary):
        print('[INFO] checkIfCentered')
        (x, y), (width, height), rotatedAngle = cv.minAreaRect(boundary)
        isUpright = height > width
        angle = 0
        if (isUpright):
            # TODO: why????
            if (rotatedAngle < 0):
                angle = 90 + rotatedAngle
            else:
                angle = rotatedAngle - 90
        else:
            angle = rotatedAngle

        return angle

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
            boundary = self.detectRDT(img)
            isCentered = False
            sizeResult = SizeResult.INVALID
            isRightOrientation = False
            angle = 0.0
            w, h = self.getBoundarySize(boundary)
            print('[INFO] width, height of boundary', w, h)

            if (w > 0 and h > 0):
                isCentered = self.checkIfCentered(boundary, img.shape, img)
                sizeResult = self.checkSize(boundary, img.shape)
                isRightOrientation = self.checkOrientation(boundary)
                angle = self.measureOrientation(boundary)
            passed = sizeResult == SizeResult.RIGHT_SIZE and isCentered and isRightOrientation
            # TODO: what does the cropRDT do? do we even need that?
            res = CaptureResult(passed, img, -1 , exposureResult, sizeResult, isCentered, isRightOrientation, isSharp, False, angle)
            print('[INFO] res', res)
            return res
        else:
            res = CaptureResult(passed, None, -1 , exposureResult, SizeResult.INVALID, False, False, isSharp, False, 0.0)
            print('[INFO] res', res)
            return res

    def cropResultWindow(self, img, boundary):
        print('[INFO] cropResultWindow started')
        h, w = self.fluRefImg.shape
        # img2 = cv.polylines(img,[np.int32(boundary)],True,(255,0,0))
        # show_image(img2)
        refBoundary = np.float32([ [0,0], [0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
        print('Refboundary', refBoundary)
        print('Boundary', boundary)
        M = cv.getPerspectiveTransform(boundary, refBoundary)
        transformedImage = cv.warpPerspective(img,M, (self.fluRefImg.shape[1], self.fluRefImg.shape[0]))
        # show_image(transformedImage)

        controlLineRect = self.checkControlLine(transformedImage)
        if (controlLineRect is None or (controlLineRect.w == 0 and controlLineRect.h == 0)):
            return (0,0)
        return (controlLineRect.w, controlLineRect.h)
    def checkControlLine(self, img):
        print('[INFO] checkControlLine')
        hls = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
        hls = cv.cvtColor(hls, cv.COLOR_RGB2HLS)
        # show_image(hls)

        kernelErode = np.ones((5,5),np.uint8)
        kernelDilate = np.ones((20,20), np.uint8)
        # Threshold the HSV image to get only blue colors
        threshold = cv.inRange(hls, CONTROL_LINE_COLOR_LOWER, CONTROL_LINE_COLOR_UPPER)
        # show_image(threshold)
        threshold = cv.erode(threshold, kernelErode,iterations = 1)
        threshold = cv.dilate(threshold,kernelDilate,iterations = 1)
        threshold = cv.GaussianBlur(threshold,(5,5),2, 2)
        # show_image(threshold)
        im2, contours, hierarchy = cv.findContours(threshold ,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        # cv.drawContours(img, contours, -1, (0,255,0), 3)
        # show_image(img)
        # show_image(im2)
        controlLineRect = None
        for contour in contours:
            # x,y,w,h = cv.boundingRect(contour)
            rect = cv.minAreaRect(contour)
            ((x, y) , (w, h), angle) = rect
            print('[INFO] boundingRect', x, y, w, h)
            # TODO: maybe ask CJ should we change the constants control line?s
            # ('[INFO] boundingRect', 1344, 0, 70, 112)
            if (CONTROL_LINE_POSITION_MIN < x and x < CONTROL_LINE_POSITION_MAX and 
                CONTROL_LINE_MIN_HEIGHT < h and CONTROL_LINE_MIN_WIDTH < w and w < CONTROL_LINE_MAX_WIDTH):
                print('[INFO] controlLine found!')
                # cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                # show_image(img)
                controlLineRect = Rect(x, y, w, h)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                cv.drawContours(img,[box],0,(0,0,255),2)
                show_image(img)

        return controlLineRect

    def interpretResult(self, src):
        print('[INFO] interpretResult')
        # colorImg = cv.imread(src, cv.IMREAD_COLOR)
        colorImg = cv.imread(src, cv.COLOR_BGR2RGB)
        img = cv.imread(src, cv.IMREAD_GRAYSCALE)
        # show_image(colorImg)

        cnt = 3
        isSizeable = SizeResult.INVALID
        isCentered = False
        isUpright = False
        boundary = None

        # TODO: what is the purpose of cnt in here? Just to ensure that it loops many time?

        while (not(isSizeable == SizeResult.RIGHT_SIZE and isCentered and isUpright) and cnt < 4):
            cnt += 1
            boundary = self.detectRDT(img, cnt)
            print('[SIFT boundary size]: ', boundary.shape)
            isSizeable = self.checkSize(boundary, img.shape)
            isCentered = self.checkIfCentered(boundary, img.shape, img)
            isUpright = self.checkOrientation(boundary)
            print("[INFO] SIFT-right size %s, center %s, orientation %s, (%.2f, %.2f), cnt %d", isSizeable, isCentered, isUpright, img.shape[0], img.shape[1], cnt)

        if (boundary.shape[0] <= 0 and boundary.shape[1] <= 0):
            return InterpretationResult()
        print('flurefimgshp,', self.fluRefImg.shape)
        print('Imgshep', img.shape)
        print('boundary', boundary)
        result = self.cropResultWindow(colorImg, boundary)
        control, testA, testB = False, False, False

        if (result[0] == 0 and result[1] == 0):
            return InterpretationResult(result, False, False, False)
        
        result = self.enhanceResultWindow(result, (5, result.shape[1]))
        # result = self.correctGamma(result, 0.75)
        # TODO: do we need to do correct Gamma?

        control = self.readControlLine(result, Point(CONTROL_LINE_POSITION, 0))
        testA = self.readTestLine(result, Point(TEST_A_LINE_POSITION, 0))
        testB = self.readTestLine(result, Point(TEST_B_LINE_POSITION))
