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
                       CONTROL_LINE_MAX_WIDTH, FIDUCIAL_MAX_WIDTH, FIDUCIAL_MIN_HEIGHT, FIDUCIAL_MIN_WIDTH,
                       FIDUCIAL_POSITION_MAX, FIDUCIAL_POSITION_MIN, FIDUCIAL_TO_CONTROL_LINE_OFFSET,
                       RESULT_WINDOW_RECT_HEIGHT, RESULT_WINDOW_RECT_WIDTH_PADDING, FIDUCIAL_COUNT,
                       FIDUCIAL_DISTANCE, ANGLE_THRESHOLD, LINE_SEARCH_WIDTH, CONTROL_LINE_POSITION, 
                       TEST_A_LINE_POSITION, TEST_B_LINE_POSITION, INTENSITY_THRESHOLD, 
                       CONTROL_INTENSITY_PEAK_THRESHOLD, TEST_INTENSITY_PEAK_THRESHOLD)
from result import (ExposureResult, CaptureResult, InterpretationResult, SizeResult)
from utils import (show_image, resize_image, Point, Rect, crop_rect)

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

    def cropResultWindow_OLD(self, img, boundary):
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
        ((x, y) , (w, h), angle) = controlLineRect
        if (controlLineRect is None or (w == 0 and h == 0)):
            return np.array()
        #For upright rect
        # crop_img = transformedImage[int(y): int(y+h),
        #                 int(x): int(x+ w)].copy()
        # show_image(crop_img)

        # For rotated rectangle
        crop_img, rotate_img = crop_rect(transformedImage, controlLineRect)
        # crop_img = getSubImage(controlLineRect, transformedImage)
        print('[INFO] crop', crop_img.shape)
        # show_image(crop_img)
        return crop_img

    def checkFiducialKMeans(self, img, K=5):
        print('[INFO] checkFiducialKmeans')
        img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
        data = img.reshape((-1,3))
        # convert to np.float32
        data = np.float32(data)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
        compactness, labels, centers = cv.kmeans(data, K, None, criteria, 10, cv.KMEANS_PP_CENTERS)
        # TODO: a plot to show how Kmeans work will be nice, sth like a 3D plot
        # sth like this: https://buzzrobot.com/dominant-colors-in-an-image-using-k-means-clustering-3c7af4622036

        # Now convert back into uint8, and make original image
        print('[INFO] KMeans shape')
        print(labels.shape)
        print(data.shape)
        print(centers.shape)

        for i in range(data.shape[0]):
            # print('Iteration: ', i)
            # int centerId = (int)labels.get(i,0)[0];
            centerId = int(labels[i][0])
            data[i][0] = centers[centerId][0]

        data = data.reshape(img.shape)
        data = data.astype(np.uint8)
        
        minCenter = ()
        minCenterVal = float('inf')

        for center in centers:
            val = center[0] + center[1] + center[2]
            if (val < minCenterVal):
                minCenter = center
                minCenterVal = val

        print('[INFO] mincenter', minCenter)
        print('[INFO] mincenterval', minCenterVal)

        thres = 0.299 * minCenter[0] + 0.587 * minCenter[1] + 0.114 * minCenter[2] + 20.0
        data = cv.cvtColor(data, cv.COLOR_RGB2GRAY)
        ret,threshold = cv.threshold(data,thres,255,cv.THRESH_BINARY_INV)
        # show_image(threshold)


        kernelErode = np.ones((5,5),np.uint8)
        kernelDilate = np.ones((20,20), np.uint8)

        threshold = cv.erode(threshold, kernelErode,iterations = 1)
        threshold = cv.dilate(threshold,kernelDilate,iterations = 1)
        threshold = cv.GaussianBlur(threshold,(5,5),2, 2)
        # show_image(threshold)
        im2, contours, hierarchy = cv.findContours(threshold ,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

        fiducialRects = []
        fiducialRect = (None, None)
        # show_image(img)
        # cv.drawContours(img, contours, -1, (0,255,0), 3)
        # show_image(img)
        
        for contour in contours:
            rect = cv.boundingRect(contour)
            x,y,w,h = rect
            rectPos = x + w
            print('[INFO] Loading contour...', rect, rectPos)
            if (FIDUCIAL_POSITION_MIN < rectPos and rectPos < FIDUCIAL_POSITION_MAX and FIDUCIAL_MIN_HEIGHT < h and FIDUCIAL_MIN_WIDTH < w and w < FIDUCIAL_MAX_WIDTH): 
                fiducialRects.append(Rect(x,y,w,h))
                print('[INFO] Found fiducial rect, ', x, y, w, h)
                # cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                # show_image(img)
        # There are some edge cases like the arrows on the right get recognized as 2 or 3 small arrows,
        # Therefore it can lead to fiducialRect > 2, but what we can do is sort them and take the average
        # between the 2 components (like the upper fiducial mean x and lower fiducial mean x)
        print('[INFO] fiducialRects', len(fiducialRects))
        if (len(fiducialRects) == FIDUCIAL_COUNT):
            center0 = fiducialRects[0].x + fiducialRects[0].w
            center1 = fiducialRects[0].x + fiducialRects[0].w

            if (len(fiducialRects) > 1):
                center1 = fiducialRects[1].x + fiducialRects[1].w

            midpoint = int((center0 + center1) / 2)
            diff = abs(center0 - center1)
            scale = 1 if FIDUCIAL_DISTANCE == 0 else diff / FIDUCIAL_DISTANCE
            offset = scale * FIDUCIAL_TO_CONTROL_LINE_OFFSET

            tl = Point(midpoint + offset - RESULT_WINDOW_RECT_HEIGHT * scale / 2.0,
                        RESULT_WINDOW_RECT_WIDTH_PADDING)
            br = Point(midpoint + offset + RESULT_WINDOW_RECT_HEIGHT * scale / 2.0,
                        img.shape[1] - RESULT_WINDOW_RECT_WIDTH_PADDING)
            img = cv.rectangle(img,(int(tl.x), int(tl.y)),(int(br.x), int(br.y)),(0,255,0),3)
            print('[INFO] tl, br', tl.x, br.x)
            # show_image(img)
            fiducialRect = (tl, br)

        print('[INFO] fiducialRect', fiducialRect)
        return fiducialRect

    def cropResultWindow(self, img, boundary):
        print('[INFO] cropResultWindow')
        h, w = self.fluRefImg.shape
        # img2 = cv.polylines(img,[np.int32(boundary)],True,(255,0,0))
        # show_image(img2)
        refBoundary = np.float32([ [0,0], [0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
        print('Refboundary', refBoundary)
        print('Boundary', boundary)
        M = cv.getPerspectiveTransform(boundary, refBoundary)
        transformedImage = cv.warpPerspective(img,M, (self.fluRefImg.shape[1], self.fluRefImg.shape[0]))
        # show_image(transformedImage)

        (tl, br) = self.checkFiducialKMeans(transformedImage)
        print('[INFO] tl, br', tl.x, tl.y, br.x, br.y)
        cropResultWindow = transformedImage[int(tl.y): int(br.y), int(tl.x):int(br.x)]
        print('[INFO] shape', cropResultWindow.shape, transformedImage.shape, img.shape)
        # show_image(cropResultWindow)
        # if (cropResultWindow.shape[0] > 0 and cropResultWindow.shape[1] > 0):
        #     cropResultWindow.reshape((RESULT_WINDOW_RECT_HEIGHT, self.fluRefImg.shape[0] - 2 * RESULT_WINDOW_RECT_WIDTH_PADDING))
        #     show_image(transformedImage)
        return cropResultWindow

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
            x,y,w,h = cv.boundingRect(contour)
            # rect = cv.minAreaRect(contour)
            # ((x, y) , (w, h), angle) = rect
            print('[INFO] boundingRect', x, y, w, h)
            # TODO: maybe ask CJ should we change the constants control line?s
            # ('[INFO] boundingRect', 1344, 0, 70, 112)
            if (CONTROL_LINE_POSITION_MIN < x and x < CONTROL_LINE_POSITION_MAX and 
                CONTROL_LINE_MIN_HEIGHT < h and CONTROL_LINE_MIN_WIDTH < w and w < CONTROL_LINE_MAX_WIDTH):
                print('[INFO] controlLine found!')
                # cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                # show_image(img)
                controlLineRect = Rect(x, y, w, h)
                # DEPRECATED ROTATED RECTANGLE WAY
                # box = cv.boxPoints(rect)
                # box = np.int0(box)
                # cv.drawContours(img,[box],0,(0,0,255),2)
                # show_image(img)

                # controlLineRect = rect

        return controlLineRect

    def enhanceResultWindow(self, controlLineRect, tile):
        return self.enhanceImage(controlLineRect, tile)

    def enhanceImage(self, img, tile):
        print('[INFO] enhanceImage')
        # show_image(img)
        result = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
        result = cv.cvtColor(result, cv.COLOR_RGB2HLS)
         # create a CLAHE object (Arguments are optional).
        clahe = cv.createCLAHE(clipLimit=10.0, tileGridSize=tile)
        channels = cv.split(result)
        cv.normalize(channels[1], channels[1], alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        cl1 = clahe.apply(channels[1])
        channels[1] = cl1
        result = cv.merge(channels)
        result = cv.cvtColor(img, cv.COLOR_HLS2RGB)
        # result = cv.cvtColor(img, cv.RGB2RGBA)
        # show_image(result)
        return result 

    def readLine(self, img, position, isControlLine):
        print('[INFO] readLine started')
        # TODO: finish this readline method
        # show_image(img)
        hls = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
        hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)

        channels = cv.split(hls)
        lower_bound = int(0 if position.x - LINE_SEARCH_WIDTH < 0 else position.x - LINE_SEARCH_WIDTH)
        upper_bound = int(position.x + LINE_SEARCH_WIDTH)

        avgIntensities = [None] * (upper_bound - lower_bound)
        avgHues = [None] * (upper_bound - lower_bound)
        avgSats = [None] * (upper_bound - lower_bound)

        min = float('inf')
        max = float('-inf')
        minIndex, maxIndex = 0,0

        # print('[INFO] hls shape', hls.shape, channels[1].shape)
        for i in range(lower_bound, upper_bound):
            sumIntensity, sumHue, sumSat = 0,0,0
            for j in range(channels[1].shape[0]):
                sumIntensity += channels[1][j][i]
                sumHue += channels[0][j][i]
                sumSat += channels[2][j][i]
            avgIntensities[i - lower_bound] = sumIntensity / channels[1].shape[0]
            avgHues[i - lower_bound] = sumHue / channels[0].shape[0]
            avgSats[i - lower_bound] = sumSat / channels[2].shape[0]
            if (avgIntensities[i - lower_bound] < min):
                min = avgIntensities[i - lower_bound]
                minIndex = i - lower_bound
            if (avgIntensities[i - lower_bound] > max):
                max = avgIntensities[i - lower_bound]
                maxIndex = i - lower_bound
        print('[INFO] min max threshold', min, max, abs(min - max))
        if (isControlLine):
            return min < INTENSITY_THRESHOLD and abs(min - max) > CONTROL_INTENSITY_PEAK_THRESHOLD
        return min < INTENSITY_THRESHOLD and abs(min - max) > TEST_INTENSITY_PEAK_THRESHOLD

    def readControlLine(self, img, controlLinePosition):
        print('[INFO] readControlLine')
        return self.readLine(img, controlLinePosition, True)

    def readTestLine(self, img, testLinePosition):
        print('[INFO] readTestLine')
        return self.readLine(img, testLinePosition, False)


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
        cv.imwrite('cropResult.png', result)
        # print('[INFO] cropResultWindow res:', result)
        control, testA, testB = False, False, False

        if (result.shape[0] == 0 and result.shape[1] == 0):
            return InterpretationResult(result, False, False, False)
        result = self.enhanceResultWindow(result, (5, result.shape[1]))
        # result = self.correctGamma(result, 0.75)
        # TODO: do we need to do correct Gamma?

        control = self.readControlLine(result, Point(CONTROL_LINE_POSITION, 0))
        testA = self.readTestLine(result, Point(TEST_A_LINE_POSITION, 0))
        testB = self.readTestLine(result, Point(TEST_B_LINE_POSITION, 0))
        print('[INFO] lines result', control, testA, testB)
        # show_image(result)
        cv.imwrite('result.png', result)
        return InterpretationResult(result, control, testA, testB)
