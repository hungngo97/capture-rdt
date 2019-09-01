from ImageProcessorScrapeReport import ImageProcessorScrapeReport
import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, default='resources/query.csv',
                        help='URLs image path')
    # This url passed in should be a list of url ( like a text file)
    args = parser.parse_args()
    imgProc = ImageProcessorScrapeReport()
    print('[INFO] start report..')
    imgProc.processFile(args.f)


if __name__ == '__main__':
    main()