from ImageProcessorScrape import ImageProcessorScrape
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from urlConstants import (
    S3_URL_BASE_PATH, TYPE, BARCODES, RDT_SCAN, ENHANCED_SCAN, MANUAL_PHOTO
)
import math

SECRET_PATH = 'keys/cough_photos_key.txt'


STATUS = 'Status'
PCR_RESULT = 'ASPREN: PCR Result'
RESULTS_USER_RESPONSE = 'Results: Shown to User Based Only on User Responses'
RDT_RESULT = 'RDT Result: What the RDT Algorithm Interpreted'
HIGH_CONTRAST_LINE_ANSWER = 'High Contrast Line Answer'
BARCODE = 'Barcode'
SECRET = ''
with open(SECRET_PATH, 'r') as file:
    SECRET = file.read()

IntepretationResultMappings = {
    'Both': 0,
    'Negative': 0,
    'Flu B': 0,
    'Flu A': 0,
    'No control line': 0
}

HighContrastLineMappings = {
    'oneLine': 0,
    'noneOfTheAbove': 0,
    'twoLines': 0,
    'threeLines': 0
}

UserResponseMappings = {
    'Negative': 0,
    'Positive': 0
}

PCRMappings = {
    'negative': 0,
    'flu A': 0,
    'flu B': 0
}


class ImageProcessorScrapeReport(ImageProcessorScrape):
    def __init__(self):
        ImageProcessorScrape.__init__(self)

    def processBarcode(self, barcode, pcr_result, results_user_response, rdt_result, high_contrast_line_answer,
                       ):
        print('[INFO] start processBarcode..')
        if barcode is None or not barcode or math.isnan(barcode):
            return None
        # Convert number barcode to string
        barcode = str(int(barcode))
        print('[INFO] processing barcode', barcode)

        URL_PATH = str(S3_URL_BASE_PATH) + \
            str(SECRET) + '/cough/' + str(barcode)
        print('[INFO] current URL path', URL_PATH)
        self.interpretResultFromURL(URL_PATH, URL_PATH)

    def processFile(self, file):
        print('[INFO] processing filename ', file)
        df = pd.read_csv(file)
        print(df.head(5))
        print("[INFO] columns")
        print(df.columns)
        print('[TESTING]')
        # print(df.iloc[0])
        # print(df['Status'][0])
        validBarcodes = 0
        total = 0
        barcodes = []

        for index, row in df.iterrows():
            # row = df.iloc[1]
            # print('YOOOO', row['Status'])
            if row[BARCODE]:
                # DEBUG
                print('[INFO] row number: ', index)
                print(BARCODE, row[BARCODE])
                print(PCR_RESULT, row[PCR_RESULT])
                print(RESULTS_USER_RESPONSE, row[RESULTS_USER_RESPONSE])
                print(RDT_RESULT, row[RDT_RESULT])
                print(HIGH_CONTRAST_LINE_ANSWER,
                      row[HIGH_CONTRAST_LINE_ANSWER])
                print(type(row[RDT_RESULT]))
                # REPORT
                validBarcodes += 1
                total += 1
                if row[RDT_RESULT] and (isinstance(row[RDT_RESULT], str) or not math.isnan(row[RDT_RESULT])):
                    IntepretationResultMappings[row[RDT_RESULT]] += 1
                if row[PCR_RESULT] and isinstance(row[PCR_RESULT], str) or not math.isnan(row[PCR_RESULT]):
                    PCRMappings[row[PCR_RESULT]] += 1
                if row[RESULTS_USER_RESPONSE]and isinstance(row[RESULTS_USER_RESPONSE], str) or not math.isnan(row[RESULTS_USER_RESPONSE]):
                    UserResponseMappings[row[RESULTS_USER_RESPONSE]] += 1
                if row[HIGH_CONTRAST_LINE_ANSWER] and isinstance(row[HIGH_CONTRAST_LINE_ANSWER], str) or not math.isnan(row[HIGH_CONTRAST_LINE_ANSWER]):
                    HighContrastLineMappings[row[HIGH_CONTRAST_LINE_ANSWER]] += 1
                # self.processBarcode(row[BARCODE], row[PCR_RESULT], row[RESULTS_USER_RESPONSE],
                #                     row[RDT_RESULT], row[HIGH_CONTRAST_LINE_ANSWER])

                # BREAK DEBUG
                # break

        print('===================REPORT===============')
        print('Total data rows: ', total)
        print('Valid barcodes: ', validBarcodes)
        print('High contrast Mappings', HighContrastLineMappings)
        print('Mappings', IntepretationResultMappings)
        print('Mappings', UserResponseMappings)
