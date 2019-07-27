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
NO_CONTROL_AREA_FOUND = 'NO_CONTROL_AREA_FOUND'
CANNOT_DETECT = 'CANNOT_DETECT'


class ImageProcessorScrape(ImageProcessor):
    def __init__(self, url):
        ImageProcessor.__init__(self, 'input/testimg8_1.jpg')
        self.url = url

    def storeEnhancedScan(self, imageFileName, barcode):
        paths = [
            ROOT_DETECTION_DIR + '/' + FALSE_SUBDIR + '/' + CANNOT_DETECT +
            '/' + str(barcode) + RDT_SCAN + '.png',
            ROOT_DETECTION_DIR + '/' + FALSE_SUBDIR + '/' + CANNOT_DETECT +
            '/' + str(barcode) + MANUAL_PHOTO + '.png',
            ROOT_DETECTION_DIR + '/' + FALSE_SUBDIR + '/' + NO_CONTROL_AREA_FOUND +
            '/' + str(barcode) + RDT_SCAN + '.png',
            ROOT_DETECTION_DIR + '/' + FALSE_SUBDIR + '/' + NO_CONTROL_AREA_FOUND +
            '/' + str(barcode) + MANUAL_PHOTO + '.png',

            ROOT_DETECTION_DIR + '/' + TRUE_SUBDIR + '/' + FLU_AB_SUBSUBDIR +
            '/' + str(barcode) + MANUAL_PHOTO + '.png',
            ROOT_DETECTION_DIR + '/' + TRUE_SUBDIR + '/' + FLU_A_SUBSUBDIR +
            '/' + str(barcode) + MANUAL_PHOTO + '.png',
            ROOT_DETECTION_DIR + '/' + TRUE_SUBDIR + '/' + FLU_B_SUBSUBDIR +
            '/' + str(barcode) + MANUAL_PHOTO + '.png',
            ROOT_DETECTION_DIR + '/' + TRUE_SUBDIR + '/' + NO_FLU_SUBSUBDIR +
            '/' + str(barcode) + MANUAL_PHOTO + '.png',

            ROOT_DETECTION_DIR + '/' + TRUE_SUBDIR + '/' + NO_FLU_SUBSUBDIR +
            '/' + str(barcode) + RDT_SCAN + '.png',
            ROOT_DETECTION_DIR + '/' + TRUE_SUBDIR + '/' + FLU_A_SUBSUBDIR +
            '/' + str(barcode) + RDT_SCAN + '.png',
            ROOT_DETECTION_DIR + '/' + TRUE_SUBDIR + '/' + FLU_B_SUBSUBDIR +
            '/' + str(barcode) + RDT_SCAN + '.png',
            ROOT_DETECTION_DIR + '/' + TRUE_SUBDIR + '/' + FLU_AB_SUBSUBDIR +
            '/' + str(barcode) + RDT_SCAN + '.png'
        ]
        for path in paths:
            print('[INFO] checking path exists', path)
            if os.path.exists(path):
                # Put it into that directory
                parts = path.split('/')
                sep = ''
                parts[-1] = str(barcode) + ENHANCED_SCAN + '.png'
                dst = sep.join(parts)

                # Save enhanced file into that directory
                print('[INFO] dst found for enhanced', dst)
                copyfile(imageFileName, dst)

    def interpretResultFromURL(self, url, barcode):
        print('[INFO] interpretResultFromURL')
        # Preprocessing
        imageFileName = readImageFromURL(url)
        if (imageFileName == 'NOT_FOUND'):
            print('[INFO] URL invalid..')
            return None
        parts = imageFileName.split('.')
        imageName = parts[0]
        imageExtension = parts[1]
        parts = imageName.split('_')
        barcode = parts[0]
        imageType = parts[1]

        if (imageType == ENHANCED_SCAN):
            # Check if this barcode is already detected
            storeEnhancedScan(self, imageFileName, barcode)
            return None

        # Detection Checking
        detectionResult = ImageProcessor.captureRDT(self, imageFileName)
        DIR_PATH = ROOT_DETECTION_DIR
        if detectionResult == None or detectionResult.sizeResult == SizeResult.INVALID:
            DIR_PATH += '/' + FALSE_SUBDIR + '/' + CANNOT_DETECT
        else:
            DIR_PATH += '/' + TRUE_SUBDIR

        # Interpret Result
        interpretResult = ImageProcessor.interpretResult(self, imageFileName)
        if (interpretResult == None):
            DIR_PATH = ROOT_DETECTION_DIR + '/' + FALSE_SUBDIR + '/' + NO_CONTROL_AREA_FOUND
        elif (interpretResult.testA and interpretResult.testB):
            DIR_PATH += '/' + FLU_AB_SUBSUBDIR
        elif (interpretResult.testA):
            DIR_PATH += '/' + FLU_A_SUBSUBDIR
        elif (interpretResult.testB):
            DIR_PATH += '/' + FLU_B_SUBSUBDIR
        else:
            DIR_PATH += '/' + NO_FLU_SUBSUBDIR

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

        # Clear things up
        os.remove(imageFileName)
