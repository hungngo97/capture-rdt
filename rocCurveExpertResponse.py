# import the necessary packages
import subprocess
from ImageProcessorScrapeReport import ImageProcessorScrapeReport
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, default='resources/query.csv',
                        help='URLs image path')
    parser.add_argument('--min', type=int, default=5,
                        help='Min variable to start')
    parser.add_argument('--max', type=int, default=40,
                        help='Max variable to stop')
    parser.add_argument('--step', type=int, default=5,
                        help='Step value between min and max')
    # This url passed in should be a list of url ( like a text file)
    args = parser.parse_args()
    for i in range(parser['min'], parser['max'], parser['step']):
        with open('variables/variables.json', 'r') as f:
            print('hahah')
    imgProc = ImageProcessorScrapeReport()
    print('[INFO] start report..')
    imgProc.processFile(args.f, args.db)


if __name__ == '__main__':
    main()
