from ImageProcessor import ImageProcessor
from ImageProcessorScrape import ImageProcessorScrape
import argparse
import sys
from urlConstants import (
    S3_URL_BASE_PATH,
    TYPE,
    BARCODES,
    RDT_SCAN,
    ENHANCED_SCAN,
    MANUAL_PHOTO,
)

SECRET_PATH = "keys/cough_photos_key.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--barcodes", type=str, default="input/barcodes.txt", help="URLs image path"
    )
    parser.add_argument("--debugCode", type=str, help="Barcode to debug")
    # This url passed in should be a list of url ( like a text file)
    args = parser.parse_args()
    imgProc = ImageProcessorScrape("")
    SECRET = ""

    # ================ DEBUG CODE ====================
    with open(SECRET_PATH, "r") as file:
        SECRET = file.read()
    barcodes = []
    if args.debugCode:
        barcode = args.debugCode
        URL_PATH = str(S3_URL_BASE_PATH) + str(SECRET) + \
            "/cough/" + str(barcode)
        print("[INFO] current URL path", URL_PATH)
        imgProc.interpretResultFromURL(URL_PATH, URL_PATH, imageType=RDT_SCAN)
        imgProc.interpretResultFromURL(
            URL_PATH, URL_PATH, imageType=MANUAL_PHOTO)
    else:
        with open(args.barcodes, "r") as file:
            # barcodes = file.read().split(",")
            barcodes = file.read().split()
        for barcode in barcodes:
            for imageType in [RDT_SCAN, MANUAL_PHOTO]:
                URL_PATH = (
                    str(S3_URL_BASE_PATH) + str(SECRET) +
                    "/cough/" + str(barcode)
                )
                print("[INFO] current URL path", URL_PATH)
                imgProc.interpretResultFromURL(
                    URL_PATH, URL_PATH, imageType=RDT_SCAN)
                # ====== TODO: add multiprocessing here to parallelize ========
                imgProc.interpretResultFromURL(
                    URL_PATH, URL_PATH, imageType=MANUAL_PHOTO)
    """
    import multiprocessing
    pool = multiprocessing.Pool()
    pool.map(function_x, list_of_files)
    """


if __name__ == "__main__":
    main()
