import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import os
import enum
import time
from constants import (OVER_EXP_THRESHOLD, UNDER_EXP_THRESHOLD, OVER_EXP_WHITE_COUNT,
                       VIEW_FINDER_SCALE_H, VIEW_FINDER_SCALE_W, SHARPNESS_THRESHOLD,
                       CROP_RATIO, MIN_MATCH_COUNT, SIZE_THRESHOLD, POSITION_THRESHOLD, ANGLE_THRESHOLD,
                       CONTROL_LINE_POSITION,
                       CONTROL_LINE_COLOR_UPPER, CONTROL_LINE_COLOR_LOWER, CONTROL_LINE_POSITION_MIN,
                       CONTROL_LINE_POSITION_MAX, CONTROL_LINE_MIN_HEIGHT, CONTROL_LINE_MIN_WIDTH,
                       CONTROL_LINE_MAX_WIDTH, FIDUCIAL_MAX_WIDTH, FIDUCIAL_MIN_HEIGHT, FIDUCIAL_MIN_WIDTH,
                       FIDUCIAL_POSITION_MAX, FIDUCIAL_POSITION_MIN, FIDUCIAL_TO_CONTROL_LINE_OFFSET,
                       RESULT_WINDOW_RECT_HEIGHT, RESULT_WINDOW_RECT_WIDTH_PADDING, FIDUCIAL_COUNT,
                       FIDUCIAL_DISTANCE, ANGLE_THRESHOLD, LINE_SEARCH_WIDTH, CONTROL_LINE_POSITION, 
                       TEST_A_LINE_POSITION, TEST_B_LINE_POSITION, INTENSITY_THRESHOLD, PEAK_HEIGHT_THRESHOLD,
                       CONTROL_INTENSITY_PEAK_THRESHOLD, TEST_INTENSITY_PEAK_THRESHOLD, DETECTION_RANGE)
from result import (ExposureResult, CaptureResult, InterpretationResult, SizeResult)
from utils import (show_image, resize_image, Point, Rect, crop_rect, peakdet)

"""
    TODO: 
    * debug 63299731, testAB, testB, testA maybe ask CJ about it (NOT DONE)
    *** F1 score and specificity (NOT TESTED)
    *** test strip boundary column, it gives you the coordinates of the detected RDT. 
     so, you dont have to run SIFT. you can just use that to those points to crop the
      RDT and correct perspective. (DONE) --> Should we keep the old boundary from python and also
      detect on that to compare with the boundary from android data? (ASK CJ)
    *** Calculate F1 Score for  "Strip Line Answer (expert)" 
    **** Generate ROC curve (first by using Excel, then using Python code)  
    **** Refactor GUI code to toggle debug for later usage
    *** Scale down the image instead of scaling up the refimage
    **** Figure is the invalid cases due to scaling thing or peak detection is failing or cropping
    # fail
    Ask CJ: Will there by any case that there is high contrast line but not user response?


"""

class ImageProcessor:
    def __init__(self):
        self.featureDetector = cv.BRISK_create(45, 4, 1)
        self.matcher = cv.BFMatcher(cv.NORM_L1, crossCheck=False)
        # Load reference image for Quickvue flu test strip
        self.fluRefImg = resize_image(
            'resources/quickvue_ref_v5.jpg', gray=True, scale_percent=100)
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
        self.siftMatcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
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

    def calculateBrightness(self, img):
        histSizeNum = 256
        histSize = 256
        histRange = (0, 256)  # the upper boundary is exclusive
        accumulate = False
        width, height = img.shape

        hist = cv.calcHist([img], [0], None, [256], [0, 256])
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
        p1 = (int(width * (1 - VIEW_FINDER_SCALE_W)/2 * 0.9),
              int(height * (1 - VIEW_FINDER_SCALE_H) / 2))
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
        return True
        #return isSharp

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
        height, width = img.shape
        p1 = (0, int(height * (1 - (VIEW_FINDER_SCALE_W) / CROP_RATIO) / 2))
        p2 = (int(width - p1[0]), int(height - p1[1]))
        #""" 
        #    ==================
        #    TODO: this mask is still HARDCODEDDDDDD!!!!!!
        #    =================
        #"""
        #p1 = (int(width * (1 - VIEW_FINDER_SCALE_W)/2 * CROP_RATIO) , 0)
        #p2 = (int(width - p1[0]), int(height - p1[1] - 65))
        roi = img[p1[1]:p2[1], p1[0]:p2[0]]
        # img = roi
        mask = np.zeros((height, width), np.uint8)
        mask[p1[1]:p2[1], p1[0]: p2[0]] = 255
        # show_image(img)
        # show_image(mask)
        # keypoints, descriptors = self.siftDetector.detectAndCompute(img, None)
        # show_image(mask)
        keypoints, descriptors = self.siftDetector.detectAndCompute(img, mask)
        print('[INFO] detect/compute time: ', time.time() - startTime)
        print('[INFO] descriptors')
        print(descriptors)
        # TODO: Find condition for this
        # if (descriptors == None or all(descriptors)):
        #     print('[WARNING] No Features on input')
        #     return None

        # Matching
        matches = self.matcher.knnMatch(self.refSiftDescriptors, descriptors, k=2)
        print('[INFO] Finish matching')
        # print('matches', matches)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.80*n.distance:
                good.append(m)


        matchingImage = cv.drawMatches(self.fluRefImg, self.refSiftKeyPoints, img,
                                       keypoints, good, None, flags=2)
        # plt.imshow(matchingImage)
        # plt.title('SIFT Brute Force matching')
        # plt.show()

        sum = 0
        distance = 0
        count = 0

        # store all the good matches as per Lowe's ratio test.
        img2 = None
        dst = None
        print('[INFO] matches')

        if len(good)> MIN_MATCH_COUNT:
            src_pts = np.float32([ self.refSiftKeyPoints[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ keypoints[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            # print('src_pts', src_pts)
            # print('dst_pst', dst_pts)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, cnt)
            print('[INFO] Finish finding Homography')
            # print('[INFO] M Matrix', M)
            # print('[INFO] mask', mask)
            matchesMask = mask.ravel().tolist()
            h,w = self.fluRefImg.shape
            pts = np.float32([ [0,0],[w-1,0],[w-1,h-1],[0,h-1] ]).reshape(-1,1,2)
            if M is None or M.size == 0:
                return None
            dst = cv.perspectiveTransform(pts,M)
            print('[INFO] dst transformation pts', dst)
            img2 = np.copy(img)
            img2 = cv.polylines(img2,[np.int32(dst)],True,(255,0,0))
            pts_box = cv.minAreaRect(dst)
            box = cv.boxPoints(pts_box) # cv2.boxPoints(rect) for OpenCV 3.x
            box = np.int0(box)
            cv.drawContours(img2,[box],0,(0,0,255),2)
            print('[INFO] finish perspective transform')
            print(box)
            # show_image(roi)
            # show_image(img2)
            # img2=None
            new_dst = np.copy(dst)
            #for i in list(range(0, 4)):
            #    min_dist = 99999999
            #    min_j = -1
            #    for j in list(range(0, 4)):
            #        print(box[i], dst[j])
            #        dist = pow(box[i][0]-dst[j][0][0], 2) + pow(box[i][1]-dst[j][0][1], 2)

            #        if dist < min_dist:
            #            print('---min', dist, j, box[i], new_dst[j])
            #            min_dist = dist
            #            min_j = j
            #    new_dst[min_j][0][0] = box[i][0]
            #    new_dst[min_j][0][1] = box[i][1]
            for i in list(range(0, 4)):
                if pts_box[2] < -45:
                    new_dst[(i+2) % 4] = [box[i]]
                else:
                    new_dst[(i+3) % 4] = [box[i]]

            dst = np.copy(new_dst)
            print(dst)

        else:
            print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            matchesMask = None
            return None

        draw_params = dict(matchColor = (255,0,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
        img3 = cv.drawMatches(self.fluRefImg,self.refSiftKeyPoints,img2,keypoints,good,None,**draw_params)
        #plt.imshow(img3, 'gray'),plt.show()
        # show_image(img3)

        h, w  = self.fluRefImg.shape
        #refBoundary = np.float32([ [0,0], [0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
        refBoundary = np.float32([ [0,0],[w-1,0],[w-1,h-1],[0,h-1] ]).reshape(-1,1,2)
        print('Refboundary', refBoundary)
        print('Boundary', dst)
        M = cv.getPerspectiveTransform(dst, refBoundary)
        print('M matrixxxx', M)
        transformedImage = cv.warpPerspective(img ,M, (self.fluRefImg.shape[1], self.fluRefImg.shape[0]))
        # show_image(roi)
        # show_image(transformedImage)

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
        h,w  = imgShape
        trueCenter = (w/2 , h/2)
        isCentered = center[0] < trueCenter[0] + (w * POSITION_THRESHOLD) and center[0] > trueCenter[0]-(w *POSITION_THRESHOLD) and center[1] < trueCenter[1]+(h *POSITION_THRESHOLD) and center[1] > trueCenter[1]-(h*POSITION_THRESHOLD)
        print('[INFO] isCentered:', imgShape)
        print('[INFO] isCentered:', trueCenter[0] + (w * POSITION_THRESHOLD), trueCenter[0]-(w *POSITION_THRESHOLD))
        print('[INFO] isCentered:', trueCenter[1] + (h * POSITION_THRESHOLD), trueCenter[1]-(h *POSITION_THRESHOLD))
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
        height,width = imgShape
        largestDimension = self.measureSize(boundary)
        print('[INFO] height', height)
        print('[INFO] largest dimension', largestDimension)
        # TODO: not sure if we even need the view finder scale??
        isRightSize = largestDimension < width * (VIEW_FINDER_SCALE_H + SIZE_THRESHOLD) and largestDimension > width * (VIEW_FINDER_SCALE_H - SIZE_THRESHOLD)
        print('[INFO] range:', width * (VIEW_FINDER_SCALE_H + SIZE_THRESHOLD), width * (VIEW_FINDER_SCALE_H - SIZE_THRESHOLD))
        
        sizeResult = SizeResult.INVALID
        if isRightSize:
            sizeResult = SizeResult.RIGHT_SIZE
        elif largestDimension > height * (VIEW_FINDER_SCALE_H + SIZE_THRESHOLD):
                sizeResult = SizeResult.LARGE
        elif largestDimension < height * (VIEW_FINDER_SCALE_H - SIZE_THRESHOLD):
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
        self.src = src
        self.img = cv.imread(src, cv.IMREAD_GRAYSCALE)
        width, height = self.img.shape
        self.height = height
        self.width = width
        img = cv.imread(src, cv.IMREAD_GRAYSCALE)
        # Check brightness
        exposureResult = (self.checkBrightness(img))
        # check sharpness (refactored)
        (x1, y1), (x2, y2) = self.getViewfinderRect(img)

        print('[INFO] top left br' , x1, y1, x2, y2)
        roi = img[x1:x2, y1:y2]
        # cropped = cv.rectangle(img,(y1, x1),(y2, x2),(0,255,0),5)
        # cropped = cv.rectangle(img,(384,0),(510,128),(0,255,0),3)
        # TODO: Fix the ROI here
        print(roi.shape)
        # show_image(img)
        # show_image(roi)
        # show_image(img)
        # show_image(cropped)
        isSharp = self.checkSharpness(roi)

        if (exposureResult == ExposureResult.NORMAL and isSharp):
            boundary = self.detectRDT(img)
            if boundary is not None:
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
                res = CaptureResult(passed, roi, -1 , exposureResult, sizeResult, isCentered, isRightOrientation, isSharp, False, angle)
                print('[INFO] res', res)
                return res

        
        res = CaptureResult(False, None, -1 , exposureResult, SizeResult.INVALID, False, False, isSharp, False, 0.0)
        print('[INFO] res', res)
        return res

    def cropResultWindow_OLD(self, img, boundary):
        print('[INFO] cropResultWindow started')
        h, w = self.fluRefImg.shape
        # img2 = cv.polylines(img,[np.int32(boundary)],True,(255,0,0))
        # show_image(img2)
        #refBoundary = np.float32([ [0,0], [0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
        refBoundary = np.float32([ [0,0],[w-1,0],[w-1,h-1],[0,h-1] ]).reshape(-1,1,2)
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
        # show_image(img)
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

        threshold = cv.erode(threshold, kernelErode)
        threshold = cv.dilate(threshold,kernelDilate)
        threshold = cv.GaussianBlur(threshold,(5,5),2, 2)
        # show_image(threshold)
        im2, contours, hierarchy = cv.findContours(threshold ,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

        fiducialRects = []
        fiducialRect = (None, None)
        arrows = 0
        fiducial = 0

        # show_image(img)
        cv.drawContours(img, contours, -1, (0,255,0), 3)
        # show_image(img)

        for contour in contours:
            rect = cv.boundingRect(contour)
            x,y,w,h = rect
            rectPos = x + w
            print('[INFO] Loading contour...', rect, rectPos)
            if (rectPos < 700):
                if (FIDUCIAL_POSITION_MIN < rectPos and rectPos < FIDUCIAL_POSITION_MAX and FIDUCIAL_MIN_HEIGHT < h and FIDUCIAL_MIN_WIDTH < w and w < FIDUCIAL_MAX_WIDTH):
                    fiducialRects.append(Rect(x,y,w,h))
                    print('[INFO] Found fiducial rect, ', x, y, w, h)
                    if x > 600:
                        #It should be the arrow
                        arrows += 1
                    if x > 250 and x < 600:
                        fiducial += 1 #should be the black rectangle
                    # cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    # show_image(img)
            else:
                if (FIDUCIAL_POSITION_MIN < rectPos and rectPos < FIDUCIAL_POSITION_MAX and 20 < h and FIDUCIAL_MIN_WIDTH < w and w < FIDUCIAL_MAX_WIDTH):
                    fiducialRects.append(Rect(x,y,w,h))
                    print('[INFO] Found fiducial rect, ', x, y, w, h)
                    if x > 600:
                        #It should be the arrow
                        arrows += 1
                    if x > 250 and x < 600:
                        fiducial += 1 #should be the black rectangle
                    # cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    # show_image(img)
        # There are some edge cases like the arrows on the right get recognized as 2 or 3 small arrows,
        # Therefore it can lead to fiducialRect > 2, but what we can do is sort them and take the average
        # between the 2 components (like the upper fiducial mean x and lower fiducial mean x)
        print('[INFO] fiducialRects', len(fiducialRects))
        print('[INFO] arrows', arrows)
        print('[INFO] black fiducial', fiducial)
        fiducialRects.sort(key=lambda rect: rect.x)
        if (len(fiducialRects) == FIDUCIAL_COUNT or (len(fiducialRects) > FIDUCIAL_COUNT and len(fiducialRects) - arrows == fiducial)):
            print('[INFO] loookkkkk', fiducialRects)
            center0 = fiducialRects[0].x + fiducialRects[0].w
            center1 = fiducialRects[1].x + fiducialRects[1].w

            if (len(fiducialRects) > 2):
                center1 = fiducialRects[-1].x + fiducialRects[-1].w

            midpoint = float(int((center0 + center1) / 2))
            diff = float(abs(center0 - center1))
            scale = 1 if FIDUCIAL_DISTANCE == 0 else diff / FIDUCIAL_DISTANCE
            offset = scale * FIDUCIAL_TO_CONTROL_LINE_OFFSET

            print('img shape', img.shape)
            print('offset', offset)
            print('CEnter0', center0)
            print('center1', center1)
            print('midpoint', midpoint)

            tl = Point(midpoint + offset - RESULT_WINDOW_RECT_HEIGHT * scale / 2.0,
                        RESULT_WINDOW_RECT_WIDTH_PADDING)
            br = Point(midpoint + offset + RESULT_WINDOW_RECT_HEIGHT * scale / 2.0,
                        img.shape[0] - RESULT_WINDOW_RECT_WIDTH_PADDING)
            img = cv.rectangle(img,(int(tl.x), int(tl.y)),(int(br.x), int(br.y)),(0,255,0),3)
            print('[INFO] tl, br', tl.x, br.x)
            # show_image(img)
            fiducialRect = (tl, br)

        print('[INFO] fiducialRect', fiducialRect)
        return fiducialRect

    def cropResultWindow(self, img, boundary):
        print('[INFO] cropResultWindow')
        # show_image(img)
        # show_image(self.fluRefImg)
        print('Boundary shape', boundary.shape)
        print('Boundary' , boundary)
        print('Image shape', img.shape)
        h, w = self.fluRefImg.shape
        # img2 = cv.polylines(img,[np.int32(boundary)],True,(255,0,0))
        # show_image(img2)
        #refBoundary = np.float32([ [0,0], [0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
        refBoundary = np.float32([ [0,0],[w-1,0],[w-1,h-1],[0,h-1] ]).reshape(-1,1,2)
        print('Refboundary', refBoundary)
        print('Boundary', boundary)
        M = cv.getPerspectiveTransform(boundary, refBoundary)
        transformedImage = cv.warpPerspective(img,M, (self.fluRefImg.shape[1], self.fluRefImg.shape[0]))
        # show_image(transformedImage)
        # TODO: this is where things went wrong, it cannot perform perspective transform

        (tl, br) = self.checkFiducialKMeans(transformedImage)
        if tl is None and br is None:
            return None
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
        print('[INFO] enhanceImage', img.shape)
        # show_image(img)
        #result = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
        result = cv.cvtColor(img, cv.COLOR_RGB2HLS)
         # create a CLAHE object (Arguments are optional).
        clahe = cv.createCLAHE(clipLimit=10.0, tileGridSize=tile)
        channels = cv.split(result)
        cv.normalize(channels[1], channels[1], alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        cl1 = clahe.apply(channels[1])
        channels[1] = cl1
        result = cv.merge(channels)
        result = cv.cvtColor(result, cv.COLOR_HLS2RGB)
        #result = cv.cvtColor(result, cv.RGB2RGBA)
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

    def detectLinesWithPeak(self, img):
        print('[INFO] start detectLinesWithPeak')
        # HSL so only take the L channel to distinguish lines
        print('[INFO] result img shape', img.shape)
        # show_image(img)
        colLightness = np.mean(img[:,:,1], axis = 0)
        # plt.plot(colLightness)
        # plt.show()
        print('[INFO] avgLightness shape', colLightness.shape)
        # Inverse the L channel so that the lines will be detected as peak, not bottom like the original array
        colLightness = [255] - colLightness
        print('[INFO] lightness shape', colLightness.shape)
        # Find peak and peak should correspond to lines
        maxtab, mintab = peakdet(colLightness, PEAK_HEIGHT_THRESHOLD)
        print('Max', maxtab)
        print('Min', mintab)
        print('Total strip count', len(maxtab))
        # plt.plot(colLightness)
        # if len(maxtab) > 0:
        #     plt.scatter(np.array(maxtab)[:,0], np.array(maxtab)[:,1], color='blue')
        # if len(mintab) > 0:
        #     plt.scatter(np.array(mintab)[:,0], np.array(mintab)[:,1], color='red')
        # plt.show()
        return maxtab, len(maxtab)

    def interpretResult(self, src, boundary=None):
        print('[INFO] interpretResult')
        self.src = src
        self.img = cv.imread(src, cv.IMREAD_GRAYSCALE)
        height, width = self.img.shape
        self.height = height
        self.width = width
        # colorImg = cv.imread(src, cv.IMREAD_COLOR)
        colorImg = cv.imread(src, cv.COLOR_BGR2RGB)
        img = cv.imread(src, cv.IMREAD_GRAYSCALE)
        # show_image(colorImg)

        cnt = 3
        isSizeable = SizeResult.INVALID
        isCentered = False
        isUpright = False
        # boundary = None

        # TODO: what is the purpose of cnt in here? Just to ensure that it loops many time?
        if boundary is None:
            #while cnt < 8:
            min_dist = 99999999
            min_boundary = boundary
            for i in list(range(3,8)):
                cnt += 1
                boundary = self.detectRDT(img, cnt)
                if boundary is None:
                    #return None
                    continue
                print('[SIFT boundary size]: ', boundary.shape)
                isSizeable = self.checkSize(boundary, img.shape)
                isCentered = self.checkIfCentered(boundary, img.shape, img)
                isUpright = self.checkOrientation(boundary)

                print("[INFO] SIFT-right size %s, center %s, orientation %s, (%.2f, %.2f), cnt %d", isSizeable, isCentered, isUpright, img.shape[0], img.shape[1], cnt)

                if isSizeable == SizeResult.RIGHT_SIZE and isCentered and isUpright:
                    print("match!!!!")
                    size = self.measureSize(boundary)
                    center = self.measureCenter(boundary, img)
                    dist_center = pow(center[0]-img.shape[1]/2.0, 2)+pow(center[1]-img.shape[0]/2.0, 2)
                    print("Dist - size", size - img.shape[0] * VIEW_FINDER_SCALE_H)
                    print("Dist - center", pow(center[0]-img.shape[1]/2.0, 2)+pow(center[1]-img.shape[0]/2.0, 2))
                    if dist_center < min_dist:
                        min_dist = dist_center
                        min_boundary = boundary

            boundary = min_boundary

        if boundary is None:
            return None

        if (boundary.shape[0] <= 0 and boundary.shape[1] <= 0):
            return InterpretationResult()
        print('flurefimgshp,', self.fluRefImg.shape)
        print('Imgshep', img.shape)
        print('boundary', boundary)

        (x1, y1), (x2, y2) = self.getViewfinderRect(img)
        print('[INFO] top left br' , x1, y1, x2, y2)
        roi = colorImg[x1:x2, y1:y2]
        # cropped = cv.rectangle(img,(y1, x1),(y2, x2),(0,255,0),5)
        # show_image(cropped)
        # show_image(roi)
        # img2 = cv.polylines(colorImg,[np.int32(boundary)],True,(255,0,0))
        # show_image(img2)
        result = self.cropResultWindow(colorImg, boundary)
        if result is None:
            return None
        # show_image(result)
        cv.imwrite('cropResult.png', result)
        # print('[INFO] cropResultWindow res:', result)
        control, testA, testB = False, False, False

        if (result.shape[0] == 0 and result.shape[1] == 0):
            return InterpretationResult(result, False, False, False)
        print('INFO - result.shape', result.shape)
        result = self.enhanceResultWindow(result, (5, result.shape[1]))
        # result = self.correctGamma(result, 0.75)
        # TODO: do we need to do correct Gamma?
        #  ===== DEBUG ======
        # show_image(result)
        # control = self.readControlLine(result, Point(CONTROL_LINE_POSITION, 0))
        # testA = self.readTestLine(result, Point(TEST_A_LINE_POSITION, 0))
        # testB = self.readTestLine(result, Point(TEST_B_LINE_POSITION, 0))
        # show_image(result)
        cv.imwrite('result.png', result)
        maxtab, numberOfLines = self.detectLinesWithPeak(result)
        for col, val, width in maxtab:
            if col > TEST_A_LINE_POSITION - DETECTION_RANGE and col < TEST_A_LINE_POSITION + DETECTION_RANGE:
                testA = True
            if col > TEST_B_LINE_POSITION - DETECTION_RANGE and col < TEST_B_LINE_POSITION + DETECTION_RANGE:
                testB = True
            if col > CONTROL_LINE_POSITION - DETECTION_RANGE and col < CONTROL_LINE_POSITION + DETECTION_RANGE:
                control = True
        with open('interpretResult.txt', 'w') as file:
            file.write(str(InterpretationResult(result, control, testA, testB, numberOfLines))) 
        print('[INFO] detection result: ', str(InterpretationResult(result, control, testA, testB, numberOfLines)))
        print('[INFO] lines result', control, testA, testB)
        return InterpretationResult(result, control, testA, testB, numberOfLines)
        
        # try:
        #     (x1, y1), (x2, y2) = self.getViewfinderRect(img)
        #     print('[INFO] top left br' , x1, y1, x2, y2)
        #     roi = colorImg[x1:x2, y1:y2]
        #     # cropped = cv.rectangle(img,(y1, x1),(y2, x2),(0,255,0),5)
        #     # show_image(cropped)
        #     # show_image(roi)
        #     result = self.cropResultWindow(colorImg, boundary)
        #     show_image(result)
        #     cv.imwrite('cropResult.png', result)
        #     # print('[INFO] cropResultWindow res:', result)
        #     control, testA, testB = False, False, False

        #     if (result.shape[0] == 0 and result.shape[1] == 0):
        #         return InterpretationResult(result, False, False, False)
        #     result = self.enhanceResultWindow(result, (5, result.shape[1]))
        #     # result = self.correctGamma(result, 0.75)
        #     # TODO: do we need to do correct Gamma?

        #     control = self.readControlLine(result, Point(CONTROL_LINE_POSITION, 0))
        #     testA = self.readTestLine(result, Point(TEST_A_LINE_POSITION, 0))
        #     testB = self.readTestLine(result, Point(TEST_B_LINE_POSITION, 0))
        #     print('[INFO] lines result', control, testA, testB)
        #     # show_image(result)
        #     cv.imwrite('result.png', result)
        #     maxtab, numberOfLines = self.detectLinesWithPeak(result)
        #     for col, val, width in maxtab:
        #         if col > TEST_A_LINE_POSITION - 10 and col < TEST_A_LINE_POSITION + 10:
        #             testA = True
        #         if col > TEST_B_LINE_POSITION - 10 and col < TEST_B_LINE_POSITION + 10:
        #             testB = True
        #         if col > CONTROL_LINE_POSITION - 10 and col < CONTROL_LINE_POSITION + 10:
        #             control = True
        #     with open('interpretResult.txt', 'w') as file:
        #         file.write(str(InterpretationResult(result, control, testA, testB, numberOfLines))) 
        #     print('[INFO] detection result: ', str(InterpretationResult(result, control, testA, testB, numberOfLines)))
        #     return InterpretationResult(result, control, testA, testB, numberOfLines)
        # except Exception as e: 
        #     # Not detected found
        #     print("Something went wrong")
        #     print(e)
        #     return None
