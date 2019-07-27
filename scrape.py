from ImageProcessor import ImageProcessor
from ImageProcessorScrape import ImageProcessorScrape
import argparse
import sys
from urlConstants import (
    S3_URL_BASE_PATH, TYPE, BARCODES, RDT_SCAN, ENHANCED_SCAN, MANUAL_PHOTO
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str,
                        help='URLs image path')
    # This url passed in should be a list of url ( like a text file)
    args = parser.parse_args()
    imgProc = ImageProcessorScrape(args.url)
    SECRET = ""
    with open('keys/cough_photos_key.txt', 'r') as file:
        SECRET = file.read()
    barcodes = []
    with open('input/barcodes.txt', 'r') as file:
        barcodes = file.read().split(",")
    for barcode in barcodes:
        for imageType in [RDT_SCAN, MANUAL_PHOTO, ENHANCED_SCAN]:
            URL_PATH = str(S3_URL_BASE_PATH) + \
                str(SECRET) + '/cough/' + str(barcode) + imageType + '.png'
            print('[INFO] current URL path', URL_PATH)
            imgProc.interpretResultFromURL(URL_PATH, URL_PATH)
    # imgProc.captureRDT(INPUT_IMAGE)
    # TODO: add multiprocessing here to parallelize


if __name__ == '__main__':
    main()
