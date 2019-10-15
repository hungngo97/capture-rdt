from ImageProcessor import ImageProcessor
from urlConstants import (
    S3_URL_BASE_PATH, TYPE, BARCODES,
    RDT_SCAN, ENHANCED_SCAN, MANUAL_PHOTO
)
from urlUtils import (
    readImageFromURL,
    extractImageFileName
)
from result import (
    ExposureResult, CaptureResult, InterpretationResult, SizeResult
)
from shutil import copyfile
import cv2 as cv
from utils import (
    show_image, clear_files, createFilePath
)
import os


"""
Detection
- True
   Interpretation
        - FluA
             - [barcode]
                 - [barcode_RDTscan].jpg
                 - [barcode_RDTscan]_cropped.jpg
                 - [barcode_RDTscan]_enhanced.jpg
                 - [barcode_EnhancedScan].jpg
             - [barcode]
                 - [Barcode_ManualPhoto].jpg
                 - [Barcode_ManualPhoto]_cropped.jpg
                 - [Barcode_ManualPhoto]_enhanced.jpg
        - FluB
        - FluAB
        - noFlu
- False (edited)

"""

ROOT_DETECTION_DIR = 'Detection'
INTERPRETATION_DIR = 'Interpretation'
TRUE_SUBDIR = 'True'
FALSE_SUBDIR = 'False'
FLU_A_SUBSUBDIR = 'FluA'
FLU_B_SUBSUBDIR = 'FluB'
FLU_AB_SUBSUBDIR = 'FluAB'
NO_FLU_SUBSUBDIR = 'noFlu'
INVALID_LINE_SUBSUBDIR = 'INVALID_LINE_SUBSUBDIR'
NO_CONTROL_AREA_FOUND = 'NO_CONTROL_AREA_FOUND'
CANNOT_DETECT = 'CANNOT_DETECT'
ENHANCED_SCAN_FROM_DEVICE_SUCCESSFUL = 'ENHANCED_SCAN_FROM_DEVICE_SUCCESSFUL'


class ImageProcessorScrape(ImageProcessor):
    def __init__(self, output_path):
        self.output_path = output_path
        ImageProcessor.__init__(self)

    def storeEnhancedScan(self, baseURL, dst):
        baseURL += ENHANCED_SCAN + '.png'
        imageFileName = readImageFromURL(baseURL)
        if (imageFileName == 'NOT_FOUND'):
            print('[INFO] No enhanced scan for this image..')
            return None
        copyfile(imageFileName, dst + '/' + imageFileName)
        os.remove(imageFileName)

    def interpretResultFromEnhancedScan(self, baseURL):
        baseURL += ENHANCED_SCAN + '.png'
        imageFileName = readImageFromURL(baseURL)
        if (imageFileName == 'NOT_FOUND'):
            print('[INFO] No enhanced scan for this image..')
            return None
        print('[INFO] using preexisting enhanced scan when no boundary found')
        img = cv.imread(imageFileName, cv.IMREAD_UNCHANGED)
        # Start intepreting from existing enhanced scan
        # Detect line location
        maxtab, numberOfLines, testAColor, testBColor, controlLineColor = self.detectLinesWithPeak(
            img)
        testA, testB, control = self.detectLinesWithRelativeLocation(maxtab)

        print('[INFO] Peak Color result from enhanced image',
              testAColor, testBColor, controlLineColor)
        # show_image(img)

        return InterpretationResult(img, controlLineColor, testAColor, testBColor, numberOfLines)

    """
        TODO: the barcode is not necessary
    """

    def interpretResultFromURL(self, baseURL, barcode, boundary=None, imageType=RDT_SCAN):
        print('[INFO] interpretResultFromURL')
        # Preprocessing
        url = baseURL + imageType + '.png'
        imageFileName = readImageFromURL(
            url, isManualPhoto=(imageType == MANUAL_PHOTO), output_path=self.output_path)
        print('[IMAGE FILE NAME]', imageFileName)
        if (imageFileName == 'NOT_FOUND'):
            print('[INFO] URL invalid..')
            return None
        parts = imageFileName.split('.')
        imageName = parts[0]
        imageExtension = parts[1]
        parts = imageName.split('_')
        barcode = parts[0]
        imageType = parts[1]
        DIR_PATH = imageType + '/' + ROOT_DETECTION_DIR

        # Detection Checking
        if boundary is None:
            detectionResult = ImageProcessor.captureRDT(
                self, imageFileName)
            if detectionResult == None:
                # if detectionResult == None or detectionResult.sizeResult == SizeResult.INVALID:
                FALSE_PATH = DIR_PATH + '/' + FALSE_SUBDIR + \
                    '/' + CANNOT_DETECT + '/' + imageName
                copyfile(imageFileName, FALSE_PATH + '/' + imageFileName)
                self.storeEnhancedScan(baseURL, DIR_PATH)
                with open(DIR_PATH + '/interpretResult.log', 'w') as f:
                    f.write(
                        str("[Error] Python cannot detect boundary in detectRDT"))
                # Try to interpret from existing enhanced scan
                interpretResult = self.interpretResultFromEnhancedScan(baseURL)
                if interpretResult is None:
                    return None
                ENHANCED_SCAN_FOUND_PATH = DIR_PATH + '/' + FALSE_SUBDIR + \
                    '/' + ENHANCED_SCAN_FROM_DEVICE_SUCCESSFUL
                createFilePath(ENHANCED_SCAN_FOUND_PATH)
                # Writing result and logs
                with open(ENHANCED_SCAN_FOUND_PATH + '/interpretResult.txt', 'w') as file:
                    file.write(str(interpretResult))
                os.remove(imageFileName)

                return interpretResult
            else:
                DIR_PATH += '/' + TRUE_SUBDIR
        else:
            DIR_PATH += '/' + TRUE_SUBDIR

        # Interpret Result
        interpretResult = ImageProcessor.interpretResult(
            self, imageFileName, boundary)

        if (interpretResult == None):
            DIR_PATH = imageType + '/' + ROOT_DETECTION_DIR + \
                '/' + FALSE_SUBDIR + '/' + NO_CONTROL_AREA_FOUND
            createFilePath(DIR_PATH)
            copyfile(imageFileName, DIR_PATH + '/' + imageFileName)
            self.storeEnhancedScan(baseURL, DIR_PATH)
            with open(DIR_PATH + '/interpretResult.log', 'w') as f:
                f.write(
                    str("[Error] Python cannot detect boundary in detectRDT"))
            # Try to interpret from existing enhanced scan
            interpretResult = self.interpretResultFromEnhancedScan(baseURL)
            if interpretResult is None:
                return None
            ENHANCED_SCAN_FOUND_PATH = imageType + '/' + ROOT_DETECTION_DIR + '/' + FALSE_SUBDIR + \
                '/' + ENHANCED_SCAN_FROM_DEVICE_SUCCESSFUL + '/' + imageFileName
            createFilePath(ENHANCED_SCAN_FOUND_PATH)
            # Writing result and logs
            with open(ENHANCED_SCAN_FOUND_PATH + '/interpretResult.txt', 'w') as file:
                print('[INFO] writing enhanced scan result')
                file.write(str(interpretResult))
            os.remove(imageFileName)

            return interpretResult
        elif (interpretResult.control and interpretResult.testA and interpretResult.testB):
            DIR_PATH += '/' + FLU_AB_SUBSUBDIR
        elif (interpretResult.control and interpretResult.testA):
            DIR_PATH += '/' + FLU_A_SUBSUBDIR
        elif (interpretResult.control and interpretResult.testB):
            DIR_PATH += '/' + FLU_B_SUBSUBDIR
        elif (interpretResult.control):
            DIR_PATH += '/' + NO_FLU_SUBSUBDIR
        else:  # Include invalid cases like only testA is true but controlLine is false etc.
            DIR_PATH += '/' + INVALID_LINE_SUBSUBDIR

        DIR_PATH += '/' + imageName

        # Create target directory & all intermediate directories if don't exists
        createFilePath(DIR_PATH)

        # Copy Result Image to desired path
        if (interpretResult != None):
            copyfile('result.png', DIR_PATH + '/' +
                     imageName + '_enhanced.png')
            copyfile('cropResult.png', DIR_PATH +
                     '/' + imageName + '_cropped.png')
            clear_files()
        copyfile(imageFileName, DIR_PATH + '/' + imageFileName)
        self.storeEnhancedScan(baseURL, DIR_PATH)
        with open(DIR_PATH + '/interpretResult.log', 'w') as f:
            f.write(str(interpretResult))

        # Clear things up
        os.remove(imageFileName)
        return interpretResult
