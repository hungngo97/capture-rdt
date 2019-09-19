# import the necessary packages
import subprocess
from ImageProcessorScrapeReport import ImageProcessorScrapeReport
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, default='resources/query.csv',
                        help='URLs image path')
    parser.add_argument('--db', type=str,
                        help='number of URLs to debug')
    # This url passed in should be a list of url ( like a text file)
    args = parser.parse_args()
    imgProc = ImageProcessorScrapeReport()
    print('[INFO] start report..')
    imgProc.processFile(args.f, args.db)


if __name__ == '__main__':
    main()
