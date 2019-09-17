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
    show_image, clear_files
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


class ImageProcessorScrape(ImageProcessor):
    def __init__(self):
        ImageProcessor.__init__(self)

    def storeEnhancedScan(self, url, dst):
        url += ENHANCED_SCAN + '.png'
        imageFileName = readImageFromURL(url)
        if (imageFileName == 'NOT_FOUND'):
            print('[INFO] No enhanced scan for this image..')
            return None
        copyfile(imageFileName, dst + '/' + imageFileName)
        os.remove(imageFileName)

    def interpretResultFromURL(self, baseURL, barcode, boundary=None):
        print('[INFO] interpretResultFromURL')
        # Preprocessing
        for imageType in [RDT_SCAN, MANUAL_PHOTO]:
            url = baseURL + imageType + '.png'
            imageFileName = readImageFromURL(
                url, isManualPhoto=(imageType == MANUAL_PHOTO))
            if (imageFileName == 'NOT_FOUND'):
                print('[INFO] URL invalid..')
                return None
            parts = imageFileName.split('.')
            imageName = parts[0]
            imageExtension = parts[1]
            parts = imageName.split('_')
            barcode = parts[0]
            imageType = parts[1]
            DIR_PATH = ROOT_DETECTION_DIR

            # Detection Checking
            if boundary is None:
                detectionResult = ImageProcessor.captureRDT(
                    self, imageFileName)
                if detectionResult == None:
                    """ TODO: I think sizeResult of detection result is not correct as well"""
                # if detectionResult == None or detectionResult.sizeResult == SizeResult.INVALID:
                    DIR_PATH += '/' + FALSE_SUBDIR + '/' + CANNOT_DETECT
                else:
                    DIR_PATH += '/' + TRUE_SUBDIR

            # Interpret Result
            interpretResult = ImageProcessor.interpretResult(
                self, imageFileName, boundary)
            # if (interpretResult == None):
            #     DIR_PATH = ROOT_DETECTION_DIR + '/' + FALSE_SUBDIR + '/' + NO_CONTROL_AREA_FOUND
            # elif (interpretResult.testA and interpretResult.testB):
            #     DIR_PATH += '/' + FLU_AB_SUBSUBDIR
            # elif (interpretResult.testA):
            #     DIR_PATH += '/' + FLU_A_SUBSUBDIR
            # elif (interpretResult.testB):
            #     DIR_PATH += '/' + FLU_B_SUBSUBDIR
            # else:
            #     DIR_PATH += '/' + NO_FLU_SUBSUBDIR

            if (interpretResult == None):
                DIR_PATH = ROOT_DETECTION_DIR + '/' + FALSE_SUBDIR + '/' + NO_CONTROL_AREA_FOUND
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
            if not os.path.exists(DIR_PATH):
                os.makedirs(DIR_PATH)
                print("Directory ", DIR_PATH,  " Created ")
            else:
                print("Directory ", DIR_PATH,  " already exists")

            # Copy Result Image to desired path
            if (interpretResult != None):
                copyfile('result.png', DIR_PATH + '/' +
                         imageName + '_enhanced.png')
                copyfile('cropResult.png', DIR_PATH +
                         '/' + imageName + '_cropped.png')
                clear_files()
            copyfile(imageFileName, DIR_PATH + '/' + imageFileName)
            self.storeEnhancedScan(baseURL, DIR_PATH)

            # Clear things up
            os.remove(imageFileName)

            return interpretResult
