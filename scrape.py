from ImageProcessor import ImageProcessor
from ImageProcessorScrape import ImageProcessorScrape
import argparse
import sys
from urlConstants import (
    S3_URL_BASE_PATH, TYPE, BARCODES, RDT_SCAN, ENHANCED_SCAN, MANUAL_PHOTO
)

SECRET_PATH = 'keys/cough_photos_key.txt'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--barcodes', type=str, default='input/barcodes.txt',
                        help='URLs image path')
    parser.add_argument('--debugCode', type=str, help='Barcode to debug')
    # This url passed in should be a list of url ( like a text file)
    args = parser.parse_args()
    imgProc = ImageProcessorScrape('')
    SECRET = ""

    # ================ DEBUG CODE ====================
    with open(SECRET_PATH, 'r') as file:
        SECRET = file.read()
    barcodes = []
    if args.debugCode:
        barcode = args.debugCode
        # barcode = '62494515' #Done, work great
        # barcode = '11111111' #Done, work great
        # barcode = '11223344' #Done, work great
        # barcode = '63512629'  # done, work great
        # barcode = '63524073' #boundary sift not correct
        # barcode = '62494549'  # done, work great
        # barcode = '62494546'  # Cannot found sift boundary well, too tight and converging boundary
        URL_PATH = str(S3_URL_BASE_PATH) + \
            str(SECRET) + '/cough/' + str(barcode)
        print('[INFO] current URL path', URL_PATH)
        imgProc.interpretResultFromURL(URL_PATH, URL_PATH)
    else:
        with open(args.barcodes, 'r') as file:
            # barcodes = file.read().split(",")
            barcodes = file.read().split()
        for barcode in barcodes:
            for imageType in [RDT_SCAN, MANUAL_PHOTO]:
                URL_PATH = str(S3_URL_BASE_PATH) + \
                    str(SECRET) + '/cough/' + str(barcode)
                print('[INFO] current URL path', URL_PATH)
                imgProc.interpretResultFromURL(URL_PATH, URL_PATH)
    # ====== TODO: add multiprocessing here to parallelize ========
    """
    import multiprocessing
    pool = multiprocessing.Pool()
    pool.map(function_x, list_of_files)
    """


if __name__ == '__main__':
    main()
