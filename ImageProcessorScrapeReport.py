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

HighContrastLineIndex = {
    'oneLine': 0,
    'noneOfTheAbove': 1,
    'twoLines': 2,
    'threeLines': 3
}

UserResponseIndex = {
    'Negative': 0,
    'Positive': 1
}

PCRMappingsIndex = {
    'negative': 0,
    'flu A': 1,
    'flu B': 2
}

IntepretationResultMappingsIndex = {
    'Both': 0,
    'Negative': 0,
    'Flu B': 0,
    'Flu A': 0,
    'No control line': 0
}


class ImageProcessorScrapeReport(ImageProcessorScrape):
    def __init__(self):
        ImageProcessorScrape.__init__(self)
        """
            Row indices correspond to the labels of highcontrast line or user response
            Column indices correspond to the labels of the interpret result.
            No interpretation = 0
            Both = 1
            testA = 2
            testB = 3
            No flu = 4
        """
        self.resultPythonComparisonWithHighContrastLineAnswer = [
            [0, 0, 0, 0, 0] for _ in range(len(HighContrastLineMappings))]
        self.resultPythonComparisonWithUserResponse = [
            [0, 0, 0, 0, 0] for _ in range(len(UserResponseMappings))]
        self.resultPythonComparisonWithPCRResult = [
            [0, 0, 0, 0, 0] for _ in range(len(PCRMappings))]
        self.resultAndroidComparisonWithHighContrastLineAnswer = [
            [0, 0, 0, 0, 0] for _ in range(len(HighContrastLineMappings))]
        self.resultAndroidComparisonWithUserResponse = [
            [0, 0, 0, 0, 0] for _ in range(len(UserResponseMappings))]
        self.resultAndroidComparisonWithPCRResult = [
            [0, 0, 0, 0, 0] for _ in range(len(PCRMappings))]
        self.lineCountResult = [
            [0, 0, 0, 0, 0] for _ in range(4)]
        self.failDetectionCount = 0
        self.failDetectionDetailList = []

    def processBarcode(self, barcode, pcr_result, results_user_response,
                       rdt_result, high_contrast_line_answer):
        print('[INFO] start processBarcode..')
        print('[PREPROCESS] barcode', barcode)
        if barcode is None or not barcode or math.isnan(barcode):
            return None
        # Convert number barcode to string
        barcode = str(int(barcode))
        print('[INFO] processing barcode', barcode)

        URL_PATH = str(S3_URL_BASE_PATH) + \
            str(SECRET) + '/cough/' + str(barcode)
        print('[INFO] current URL path', URL_PATH)
        interpretResult = self.interpretResultFromURL(URL_PATH, URL_PATH)
        if interpretResult is None:
            # ===== TODO: Do something else if interpretResult is None
            self.failDetectionCount += 1
            failDetectionDetail = {
                "barcode": barcode,
                "url": URL_PATH,
                "pcrResult": pcr_result,
                "resultsUserResponse": results_user_response,
                "rdtResult": rdt_result,
                "HighcontrastLineAnswer": high_contrast_line_answer
            }
            self.failDetectionDetailList.append(failDetectionDetail)
            return None
        if pcr_result and (isinstance(pcr_result, str) or not math.isnan(pcr_result)):
            self.comparePCRResult(interpretResult, pcr_result)
        if results_user_response and (isinstance(results_user_response, str) or not math.isnan(results_user_response)):
            self.compareUserResponse(interpretResult, results_user_response)
        if high_contrast_line_answer and (isinstance(high_contrast_line_answer, str) or not math.isnan(high_contrast_line_answer)):
            self.compareHighContrastLine(
                interpretResult, high_contrast_line_answer)
        if high_contrast_line_answer and (isinstance(high_contrast_line_answer, str) or not math.isnan(high_contrast_line_answer)):
            self.compareLineCount(interpretResult, high_contrast_line_answer)
        if (high_contrast_line_answer and (isinstance(high_contrast_line_answer, str) or not math.isnan(high_contrast_line_answer))) and (results_user_response and (isinstance(results_user_response, str) or not math.isnan(results_user_response))):
            self.compareAndroidResult(
                rdt_result, pcr_result, results_user_response, high_contrast_line_answer)

    def compareAndroidResult(self, rdt_result, pcr_result, results_user_response, high_contrast_line_answer):
        print('[INFO] start compareAndroidResult')
        contrastLineRowIndex = HighContrastLineIndex[high_contrast_line_answer]
        pcrRowIndex = PCRMappingsIndex[pcr_result]
        userResponseRowIndex = UserResponseIndex[results_user_response]
        rdtResultColumnIndex = IntepretationResultMappingsIndex[rdt_result]

        self.resultAndroidComparisonWithHighContrastLineAnswer[
            contrastLineRowIndex][rdtResultColumnIndex] += 1
        self.resultAndroidComparisonWithPCRResult[pcrRowIndex][rdtResultColumnIndex] += 1
        self.resultAndroidComparisonWithUserResponse[userResponseRowIndex][rdtResultColumnIndex] += 1

    def compareLineCount(self, interpretResult, high_contrast_line_answer):
        print('[INFO] start compareLineCount')
        row_index = HighContrastLineIndex[high_contrast_line_answer]
        if interpretResult.lineCount:
            self.lineCountResult[row_index][interpretResult.lineCount] += 1

    def comparePCRResult(self, interpretResult, pcr_result):
        print('[INFO] start comparePCRResult')
        row_index = PCRMappingsIndex[pcr_result]
        col_index = 0
        if (interpretResult == None):
            col_index += 0
        elif (interpretResult.testA and interpretResult.testB):
            col_index += 1
        elif (interpretResult.testA):
            col_index += 2
        elif (interpretResult.testB):
            col_index += 3
        else:
            col_index += 4
        print('[INFO] row , column indices: ', row_index, col_index)
        self.resultPythonComparisonWithPCRResult[row_index][col_index] += 1

    def compareUserResponse(self, interpretResult, results_user_response):
        print('[INFO] start comparePCRResult')
        row_index = UserResponseIndex[results_user_response]
        col_index = 0
        if (interpretResult == None):
            col_index += 0
        elif (interpretResult.testA and interpretResult.testB):
            col_index += 1
        elif (interpretResult.testA):
            col_index += 2
        elif (interpretResult.testB):
            col_index += 3
        else:
            col_index += 4
        print('[INFO] row , column indices: ', row_index, col_index)
        self.resultPythonComparisonWithUserResponse[row_index][col_index] += 1

    def compareHighContrastLine(self, interpretResult, high_contrast_line_answer):
        print('[INFO] start comparePCRResult')
        row_index = HighContrastLineIndex[high_contrast_line_answer]
        col_index = 0
        if (interpretResult == None):
            col_index += 0
        elif (interpretResult.testA and interpretResult.testB):
            col_index += 1
        elif (interpretResult.testA):
            col_index += 2
        elif (interpretResult.testB):
            col_index += 3
        else:
            col_index += 4
        print('[INFO] row , column indices: ', row_index, col_index)
        self.resultPythonComparisonWithHighContrastLineAnswer[row_index][col_index] += 1

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
        # ================ DEBUG =========================
        DEBUG_AMOUNT = 10
        DEBUG_counter = 1

        for index, row in df.iterrows():
            # row = df.iloc[DEBUG_counter]
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
                self.processBarcode(row[BARCODE], row[PCR_RESULT], row[RESULTS_USER_RESPONSE],
                                    row[RDT_RESULT], row[HIGH_CONTRAST_LINE_ANSWER])

                # BREAK DEBUG
                DEBUG_counter += 1
                if (DEBUG_counter > DEBUG_AMOUNT):
                    break

        print('+++++++++++++++++++++++++++++++REPORT+++++++++++++++++++++++++++++++++++++')
        print('----------------------------Overall Statistics----------------------------')
        print('Total data rows: ', total)
        print('Valid barcodes: ', validBarcodes)
        print('High contrast Mappings', HighContrastLineMappings)
        print('Interpretation Result Mappings', IntepretationResultMappings)
        print('User Response Mappings', UserResponseMappings)
        print('--------------------------Accuracy Table Comparison-----------------------')
        print('=======Python Result========')
        print('High Contrast Line Result Table',
              self.resultPythonComparisonWithHighContrastLineAnswer)
        print('PCR Result Table', self.resultPythonComparisonWithPCRResult)
        print('User Response Result Table',
              self.resultPythonComparisonWithUserResponse)
        self.reportPythonResultStatistics()
        print('=======Android Result=========')
        print('High Contrast Line Result Table',
              self.resultAndroidComparisonWithHighContrastLineAnswer)
        print('PCR Result Table', self.resultAndroidComparisonWithPCRResult)
        print('User Response Result Table',
              self.resultAndroidComparisonWithUserResponse)
        self.reportAndroidResultStatistics()
        print('=======Line Count=======')
        print('Line count table', self.lineCountResult)
        print('~~~~~~~')
        self.reportLineCountStatistics()

    def reportPythonResultStatistics(self):
        # High Contrast Line
        # ========= TODO: FINISH CALCULATING THIS ACCURACY PERCENTAGE ======
        # PCR Result

        # User Reponse
        print('')

    def reportAndroidResultStatistics(self):
        # High Contrast Line

        # PCR Result

        # User Reponse
        print('')

    def reportLineCountStatistics(self):
        totalCorrect = 0
        lines = 0
        for row, line in enumerate(self.lineCountResult):
            totalCorrect += self.lineCountResult[row][row]
            totalCorrectInCurrentRow = self.lineCountResult[row][row]
            totalInCurrentRow = 0
            for col, num in enumerate(line):
                totalInCurrentRow += num
                lines += num
            print('True Label (Number of line):  ', row)
            print('Number of correct prediction: ', totalCorrectInCurrentRow)
            print('Percentage Accuracy: ',
                  totalCorrectInCurrentRow / totalInCurrentRow if totalCorrectInCurrentRow != 0 else 'N/A')
            print('~~~~~~~~')

        print('Total number of lines data: ', lines)
        print('Number of correct prediction: ', totalCorrect)
        print('Percentage correct: ', totalCorrect /
              lines if lines != 0 else 'N/A')
        print('~~~~~~~~~~~~')
