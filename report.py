from ImageProcessorScrapeReport import ImageProcessorScrapeReport
import argparse
import sys
from urlConstants import (
    RDT_SCAN, ENHANCED_SCAN, MANUAL_PHOTO
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, default='resources/query.csv',
                        help='URLs image path')
    parser.add_argument('--db', type=str,
                        help='number of URLs to debug')
    parser.add_argument('--imageType', type=str, default='RDTScan',
                        help='Image type to process')
    # This url passed in should be a list of url ( like a text file)
    args = parser.parse_args()
    imgProc = ImageProcessorScrapeReport('')
    print('[INFO] start report..')
    imgProc.processFile(args.f, args.db, imageType='_' + args.imageType)


if __name__ == '__main__':
    main()
