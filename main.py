from ImageProcessor import ImageProcessor
import argparse
import sys
# INPUT_IMAGE = 'input/testimg10_1.jpg'
# # 4.1 doesnot work well because the arrow got cut in half
# # 10.1 doesnot work well


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str,
                        help='RDT image file path')
    args = parser.parse_args()
    imgProc = ImageProcessor()
    # imgProc.captureRDT(INPUT_IMAGE)
    imgProc.interpretResult(args.f)


if __name__ == '__main__':
    main()
