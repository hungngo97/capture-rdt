from __future__ import division
from ImageProcessorScrape import ImageProcessorScrape
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from urlConstants import (
    S3_URL_BASE_PATH, TYPE, BARCODES, RDT_SCAN, ENHANCED_SCAN, MANUAL_PHOTO
)
from utils import (
    calculateF1Score, calculatePrecisionScore, calculateRecallScore
)
import math
import json
import sys
"""
    TODO: Figure the peak line detection + why python results is different from android
"""

SECRET_PATH = 'keys/cough_photos_key.txt'
STATUS = 'Status'
PCR_RESULT = 'ASPREN: PCR Result'
RESULTS_USER_RESPONSE = 'Results: Shown to User Based Only on User Responses'
RDT_RESULT = 'RDT Result: What the RDT Algorithm Interpreted'
HIGH_CONTRAST_LINE_ANSWER = 'High Contrast Line Answer'
TEST_STRIP_BOUNDARY = 'Test Strip Boundary'
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
    'noneOfTheAbove': 0,
    'oneLine': 1,
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
    'Negative': 1,
    'Flu B': 2,
    'Flu A': 3,
    'No control line': 4
}


class ImageProcessorScrapeReport(ImageProcessorScrape):
    def __init__(self):
        self.colLabels = ['No interpretation', 'Both', 'Test A', 'Test B', 'No Flu']
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

    def preprocessTestStripBoundary(self, test_strip_boundary):
        print('[INFO] teststripboundary', test_strip_boundary)
        boundary = json.loads(test_strip_boundary)
        print('[INFO] Test Strip Boundary JSON', test_strip_boundary)
        print('[INFO] Test strip map conversion', boundary)
        print(boundary[0]["x"])
        arr = []
        for point in boundary:
            print(point)
            x = point["x"]
            y = point["y"]
            arr.append([[x, y]])
        print('[INFO] arr JSON', arr)
        print('[INFO] JSON result', np.array(arr))
        return np.array(arr).astype('float32')


    def processBarcode(self, barcode, pcr_result, results_user_response,
                       rdt_result, high_contrast_line_answer, test_strip_boundary):
        print('[INFO] start processBarcode..')
        print('[PREPROCESS] barcode', barcode)
        if barcode is None or not barcode or math.isnan(barcode):
            return None
        # Convert number barcode to string
        barcode = str(int(barcode))
        boundary = None
        # convert test_strip_boundary to dictionary
        if test_strip_boundary and (isinstance(test_strip_boundary, str) or not math.isnan(test_strip_boundary)):
            boundary = self.preprocessTestStripBoundary(test_strip_boundary)
            print('[INFO] boundary preprocessing result', boundary)
        print('[INFO] processing barcode', barcode)

        URL_PATH = str(S3_URL_BASE_PATH) + \
            str(SECRET) + '/cough/' + str(barcode)
        print('[INFO] current URL path', URL_PATH)
        interpretResult = self.interpretResultFromURL(URL_PATH, URL_PATH, boundary)
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
        self.compareAndroidResult(
            rdt_result, pcr_result, results_user_response, high_contrast_line_answer)

    def compareAndroidResult(self, rdt_result, pcr_result, results_user_response, high_contrast_line_answer):
        print('[INFO] start compareAndroidResult')
        # if not (high_contrast_line_answer and (isinstance(high_contrast_line_answer, str) or not math.isnan(high_contrast_line_answer))) or \
        #     not (pcr_result and (isinstance(pcr_result, str) or not math.isnan(pcr_result))) or \
        #     not (results_user_response and (isinstance(results_user_response, str) or not math.isnan(results_user_response))):
        #     return None
        rdtResultColumnIndex = IntepretationResultMappingsIndex[rdt_result]

        if (high_contrast_line_answer and (isinstance(high_contrast_line_answer, str) or not math.isnan(high_contrast_line_answer))):
            contrastLineRowIndex = HighContrastLineIndex[high_contrast_line_answer]
            self.resultAndroidComparisonWithHighContrastLineAnswer[
                contrastLineRowIndex][rdtResultColumnIndex] += 1


        if (pcr_result and (isinstance(pcr_result, str) or not math.isnan(pcr_result))):
            pcrRowIndex = PCRMappingsIndex[pcr_result]
            self.resultAndroidComparisonWithPCRResult[pcrRowIndex][rdtResultColumnIndex] += 1

        if (results_user_response and (isinstance(results_user_response, str) or not math.isnan(results_user_response))):
            userResponseRowIndex = UserResponseIndex[results_user_response]
            self.resultAndroidComparisonWithUserResponse[userResponseRowIndex][rdtResultColumnIndex] += 1

    def compareLineCount(self, interpretResult, high_contrast_line_answer):
        print('[INFO] start compareLineCount')
        print(interpretResult.lineCount)
        row_index = HighContrastLineIndex[high_contrast_line_answer]
        if interpretResult.lineCount:
            self.lineCountResult[row_index][interpretResult.lineCount] += 1
        else:
            self.lineCountResult[row_index][0] += 1

    def comparePCRResult(self, interpretResult, pcr_result):
        print('[INFO] start comparePCRResult')
        row_index = PCRMappingsIndex[pcr_result]
        col_index = 0
        # if (interpretResult == None):
        #     col_index += 0
        # elif (interpretResult.testA and interpretResult.testB):
        #     col_index += 1
        # elif (interpretResult.testA):
        #     col_index += 2
        # elif (interpretResult.testB):
        #     col_index += 3
        # else:
        #     col_index += 4

        if (interpretResult == None):
            col_index += 0
        elif (interpretResult.control and interpretResult.testA and interpretResult.testB):
            col_index += 1
        elif (interpretResult.control and interpretResult.testA):
            col_index += 2
        elif (interpretResult.control and interpretResult.testB):
            col_index += 3
        elif (interpretResult.control):
            col_index += 4
        else:  # Include invalid cases like only testA is true but controlLine is false etc.
            col_index += 0

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

    def processFile(self, file, db):
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
        DEBUG_AMOUNT = int(db) if db else None
        DEBUG_counter = 1

        for index, row in df.iterrows():
            # row = df.iloc[DEBUG_counter]
            # try:
                if row[BARCODE]:
                    # DEBUG
                    print('[INFO] row number: ', index)
                    print(BARCODE, row[BARCODE])
                    print(PCR_RESULT, row[PCR_RESULT])
                    print(RESULTS_USER_RESPONSE, row[RESULTS_USER_RESPONSE])
                    print(RDT_RESULT, row[RDT_RESULT])
                    print(HIGH_CONTRAST_LINE_ANSWER,
                        row[HIGH_CONTRAST_LINE_ANSWER])
                    print(TEST_STRIP_BOUNDARY, row[TEST_STRIP_BOUNDARY])
                    print(type(row[RDT_RESULT]))
                    # REPORT
                    validBarcodes += 1
                    if row[RDT_RESULT] and (isinstance(row[RDT_RESULT], str) or not math.isnan(row[RDT_RESULT])):
                        IntepretationResultMappings[row[RDT_RESULT]] += 1
                    if row[PCR_RESULT] and isinstance(row[PCR_RESULT], str) or not math.isnan(row[PCR_RESULT]):
                        PCRMappings[row[PCR_RESULT]] += 1
                    if row[RESULTS_USER_RESPONSE]and isinstance(row[RESULTS_USER_RESPONSE], str) or not math.isnan(row[RESULTS_USER_RESPONSE]):
                        UserResponseMappings[row[RESULTS_USER_RESPONSE]] += 1
                    if row[HIGH_CONTRAST_LINE_ANSWER] and isinstance(row[HIGH_CONTRAST_LINE_ANSWER], str) or not math.isnan(row[HIGH_CONTRAST_LINE_ANSWER]):
                        HighContrastLineMappings[row[HIGH_CONTRAST_LINE_ANSWER]] += 1
                    self.processBarcode(row[BARCODE], row[PCR_RESULT], row[RESULTS_USER_RESPONSE],
                                        row[RDT_RESULT], row[HIGH_CONTRAST_LINE_ANSWER], row[TEST_STRIP_BOUNDARY])

                    # BREAK DEBUG
                    DEBUG_counter += 1
                    if (db and DEBUG_counter > DEBUG_AMOUNT):
                        break
                total += 1
            # except: 
            #     continue
        print('+++++++++++++++++++++++++++++++REPORT+++++++++++++++++++++++++++++++++++++')
        print('----------------------------Overall Statistics----------------------------')
        print('Total data rows: ', total)
        print('Valid barcodes: ', validBarcodes)
        print('High contrast Mappings', HighContrastLineMappings)
        print('Interpretation Result Mappings', IntepretationResultMappings)
        print('User Response Mappings', UserResponseMappings)
        print('--------------------------Accuracy Table Comparison-----------------------')
        print('=======Python Result========')
        print('High Contrast Line Result Table')
        self.printTable(HighContrastLineIndex.keys, self.colLabels, self.resultPythonComparisonWithHighContrastLineAnswer)
        print('PCR Result Table')
        self.printTable(PCRMappings.keys, self.colLabels, self.resultPythonComparisonWithPCRResult)
        print('User Response Result Table')
        self.printTable(UserResponseMappings.keys, self.colLabels, self.resultPythonComparisonWithUserResponse)
        self.reportPythonResultStatistics()
        print('=======Android Result=========')
        print('High Contrast Line Result Table')
        self.printTable(HighContrastLineIndex.keys, self.colLabels, self.resultAndroidComparisonWithHighContrastLineAnswer)
        print('PCR Result Table')
        self.printTable(PCRMappings.keys, self.colLabels, self.resultAndroidComparisonWithPCRResult)
        print('User Response Result Table')
        self.printTable(HighContrastLineIndex.keys, self.colLabels, self.resultAndroidComparisonWithUserResponse)
        self.reportAndroidResultStatistics()
        print('=======Line Count=======')
        print('Line count table')
        self.printTable(range(4), range(5), self.lineCountResult)
        print('~~~~~~~')
        self.reportLineCountStatistics()

    def reportPythonResultStatistics(self):
        # Column
        #         No interpretation = 0
        # Both = 1
        # testA = 2
        # testB = 3
        # No flu = 4
        # High Contrast Line
        # ROw
        #  'oneLine': 1,
        # 'noneOfTheAbove': 0,
        # 'twoLines': 2,
        # 'threeLines': 3
        totalCorrect = 0
        lines = 0
        result = []
        currResult = {}
        for correctContrastLineNumber, row in enumerate(self.resultPythonComparisonWithHighContrastLineAnswer):
            totalCorrectInCurrentRow = 0
            totalInCurrentRow = 0
            for i, num in enumerate(row):
                # TODO: not sure if this is correct. Ask CJ!
                if (correctContrastLineNumber == 0 and i == 0) or \
                    ((correctContrastLineNumber == 1 and i == 4)) or \
                    (correctContrastLineNumber == 2 and (i == 2 or i == 3)) or \
                        (correctContrastLineNumber == 3 and i == 1):
                    totalCorrect += num
                    totalCorrectInCurrentRow += num
                lines += num
                totalInCurrentRow += num
            print('True Label (Contrast Line Number):  ',
                                    {v: k for k, v in HighContrastLineIndex.items()}[correctContrastLineNumber])
            print('Number of correct prediction: ', totalCorrectInCurrentRow)
            print('Total number of Labels: ', totalInCurrentRow)
            print('Percentage Accuracy: ',
                  totalCorrectInCurrentRow / totalInCurrentRow if totalInCurrentRow != 0 else 'N/A')
            print('~~~~~~~~')
            currResult[{v: k for k, v in HighContrastLineIndex.items()}[correctContrastLineNumber]] = {
                'androidResult' : totalCorrectInCurrentRow,
                'result': totalInCurrentRow,
                'accuracy': totalCorrectInCurrentRow / totalInCurrentRow if totalInCurrentRow != 0 else 'N/A'
            }

        print('Total number of high contrast line data: ', lines)
        print('Number of correct prediction: ', totalCorrect)
        print('Percentage correct: ', totalCorrect /
              lines if lines != 0 else 'N/A')
        print('~~~~~~~~~~~~')
        currResult['overall'] = {
            'androidResult': totalCorrect,
            'result': lines,
            'accuracy': totalCorrect /
              lines if lines != 0 else 'N/A'
        }
        result.append({'High Contrast': currResult})
                # Column
        #         No interpretation = 0
        # Both = 1
        # testA = 2
        # testB = 3
        # No flu = 4
        # PCR Result
            #   'negative': 0,
            # 'flu A': 1,
            # 'flu B': 2
        totalCorrect = 0
        lines = 0
        currResult = {}
        for correctLabel, row in enumerate(self.resultPythonComparisonWithPCRResult):
            totalCorrectInCurrentRow = 0
            totalInCurrentRow = 0
            for i, num in enumerate(row):
                # TODO: not sure if this is correct. Ask CJ!
                if (correctLabel == 0 and (i == 4 or i == 0)) or \
                    ((correctLabel == 1 and (i == 2 or i == 1))) or \
                    (correctLabel == 2 and (i == 1 or i == 3)):
                    totalCorrect += num
                    totalCorrectInCurrentRow += num
                lines += num
                totalInCurrentRow += num
            print('True Label (PCR Result):  ',
                  {v: k for k, v in PCRMappingsIndex.items()}[correctLabel])
            print('Number of correct prediction: ', totalCorrectInCurrentRow)
            print('Total number of Labels: ', totalInCurrentRow)
            print('Percentage Accuracy: ',
                  totalCorrectInCurrentRow / totalInCurrentRow if totalInCurrentRow != 0 else 'N/A')
            print('~~~~~~~~')
            currResult[{v: k for k, v in PCRMappingsIndex.items()}[correctLabel]] = {
                'androidResult' : totalCorrectInCurrentRow,
                'result': totalInCurrentRow,
                'accuracy': totalCorrectInCurrentRow / totalInCurrentRow if totalInCurrentRow != 0 else 'N/A'
            }

        print('Total number of PCR Result data: ', lines)
        print('Number of correct prediction: ', totalCorrect)
        print('Percentage correct: ', totalCorrect /
              lines if lines != 0 else 'N/A')
        print('~~~~~~~~~~~~')
        currResult['overall'] = {
            'androidResult': totalCorrect,
            'result': lines,
            'accuracy': totalCorrect /
              lines if lines != 0 else 'N/A'
        }
        result.append({'PCR': currResult})
        # User Reponse
        # 'Negative': 0,
        # 'Positive': 1
        totalCorrect = 0
        lines = 0
        currResult = {}
        for correctLabel, row in enumerate(self.resultPythonComparisonWithUserResponse):
            totalCorrectInCurrentRow = 0
            totalInCurrentRow = 0
            for i, num in enumerate(row):
                # TODO: not sure if this is correct. Ask CJ!
                if (correctLabel == 0 and (i == 4)) or \
                    ((correctLabel == 1 and (i != 0 and i != 4))):
                    totalCorrect += num
                    totalCorrectInCurrentRow += num
                lines += num
                totalInCurrentRow += num
            print('True Label (User Response Result):  ',
                  {v: k for k, v in UserResponseIndex.items()}[correctLabel])
            print('Number of correct prediction: ', totalCorrectInCurrentRow)
            print('Total number of Labels: ', totalInCurrentRow)
            print('Percentage Accuracy: ',
                  totalCorrectInCurrentRow / totalInCurrentRow if totalInCurrentRow != 0 else 'N/A')
            print('~~~~~~~~')
            currResult[{v: k for k, v in UserResponseIndex.items()}[correctLabel]] = {
                'androidResult' : totalCorrectInCurrentRow,
                'result': totalInCurrentRow,
                'accuracy': totalCorrectInCurrentRow / totalInCurrentRow if totalInCurrentRow != 0 else 'N/A'
            }

        print('Total number of PCR Result data: ', lines)
        print('Number of correct prediction: ', totalCorrect)
        print('Percentage correct: ', totalCorrect /
              lines if lines != 0 else 'N/A')
        print('~~~~~~~~~~~~')
        currResult['overall'] = {
            'androidResult': totalCorrect,
            'result': lines,
            'accuracy': totalCorrect /
              lines if lines != 0 else 'N/A'
        }
        result.append({'User Response': currResult})
        self.printF1ScoreUserResponse(currResult)
        self.printResultTable(result)

    def reportAndroidResultStatistics(self):
        # Column
        #         No interpretation = 0
        # Both = 1
        # testA = 2
        # testB = 3
        # No flu = 4
        # High Contrast Line
        # ROw
        #  'oneLine': 1,
        # 'noneOfTheAbove': 0,
        # 'twoLines': 2,
        # 'threeLines': 3
        totalCorrect = 0
        lines = 0
        result = []
        currResult = {}
        for correctContrastLineNumber, row in enumerate(self.resultAndroidComparisonWithHighContrastLineAnswer):
            totalCorrectInCurrentRow = 0
            totalInCurrentRow = 0
            for i, num in enumerate(row):
                # TODO: not sure if this is correct. Ask CJ!
                if (correctContrastLineNumber == 0 and i == 0) or \
                    ((correctContrastLineNumber == 1 and i == 4)) or \
                    (correctContrastLineNumber == 2 and (i == 2 or i == 3)) or \
                        (correctContrastLineNumber == 3 and i == 1):
                    totalCorrect += num
                    totalCorrectInCurrentRow += num
                lines += num
                totalInCurrentRow += num
            # print(list(HighContrastLineIndex.keys()))
            print('True Label (Contrast Line Number):  ',
                  {v: k for k, v in HighContrastLineIndex.items()}[correctContrastLineNumber])
        
            print('Number of correct prediction: ', totalCorrectInCurrentRow)
            print('Total number of Labels: ', totalInCurrentRow)
            print('Percentage Accuracy: ',
                  totalCorrectInCurrentRow / totalInCurrentRow if totalInCurrentRow != 0 else 'N/A')
            print('~~~~~~~~')
            currResult[{v: k for k, v in HighContrastLineIndex.items()}[correctContrastLineNumber]] = {
                'androidResult' : totalCorrectInCurrentRow,
                'result': totalInCurrentRow,
                'accuracy': totalCorrectInCurrentRow / totalInCurrentRow if totalInCurrentRow != 0 else 'N/A'
            }

        print('Total number of high contrast line data: ', lines)
        print('Number of correct prediction: ', totalCorrect)
        print('Percentage correct: ', totalCorrect /
              lines if lines != 0 else 'N/A')
        print('~~~~~~~~~~~~')
        currResult['overall'] = {
            'androidResult': totalCorrect,
            'result': lines,
            'accuracy': totalCorrect /
              lines if lines != 0 else 'N/A'
        }
        result.append({'High Contrast': currResult})
                # Column
        #         No interpretation = 0
        # Both = 1
        # testA = 2
        # testB = 3
        # No flu = 4
        # PCR Result
            #   'negative': 0,
            # 'flu A': 1,
            # 'flu B': 2
        totalCorrect = 0
        lines = 0
        currResult = {}
        for correctLabel, row in enumerate(self.resultAndroidComparisonWithPCRResult):
            totalCorrectInCurrentRow = 0
            totalInCurrentRow = 0
            for i, num in enumerate(row):
                # TODO: not sure if this is correct. Ask CJ!
                if (correctLabel == 0 and (i == 4 or i == 0)) or \
                    ((correctLabel == 1 and (i == 2 or i == 1))) or \
                    (correctLabel == 2 and (i == 1 or i == 3)):
                    totalCorrect += num
                    totalCorrectInCurrentRow += num
                lines += num
                totalInCurrentRow += num
            print('True Label (PCR Result):  ',
                  {v: k for k, v in PCRMappingsIndex.items()}[correctLabel])
            print('Number of correct prediction: ', totalCorrectInCurrentRow)
            print('Total number of Labels: ', totalInCurrentRow)
            print('Percentage Accuracy: ',
                  totalCorrectInCurrentRow / totalInCurrentRow if totalInCurrentRow != 0 else 'N/A')
            print('~~~~~~~~')
            currResult[{v: k for k, v in PCRMappingsIndex.items()}[correctLabel]] = {
                'androidResult' : totalCorrectInCurrentRow,
                'result': totalInCurrentRow,
                'accuracy': totalCorrectInCurrentRow / totalInCurrentRow if totalInCurrentRow != 0 else 'N/A'
            }

        print('Total number of PCR Result data: ', lines)
        print('Number of correct prediction: ', totalCorrect)
        print('Percentage correct: ', totalCorrect /
              lines if lines != 0 else 'N/A')
        print('~~~~~~~~~~~~')
        currResult['overall'] = {
            'androidResult': totalCorrect,
            'result': lines,
            'accuracy': totalCorrect /
              lines if lines != 0 else 'N/A'
        }
        result.append({'PCR': currResult})
        # User Reponse
        # 'Negative': 0,
        # 'Positive': 1
        totalCorrect = 0
        lines = 0
        currResult = {}

        for correctLabel, row in enumerate(self.resultAndroidComparisonWithUserResponse):
            totalCorrectInCurrentRow = 0
            totalInCurrentRow = 0
            for i, num in enumerate(row):
                # TODO: not sure if this is correct. Ask CJ!
                if (correctLabel == 0 and (i == 4)) or \
                    ((correctLabel == 1 and (i != 4))):
                    totalCorrect += num
                    totalCorrectInCurrentRow += num
                lines += num
                totalInCurrentRow += num
            print('True Label (User Response Result):  ',
                  {v: k for k, v in UserResponseIndex.items()}[correctLabel])
            print('Number of correct prediction: ', totalCorrectInCurrentRow)
            print('Total number of Labels: ', totalInCurrentRow)
            print('Percentage Accuracy: ',
                  totalCorrectInCurrentRow / totalInCurrentRow if totalInCurrentRow != 0 else 'N/A')
            print('~~~~~~~~')
            currResult[{v: k for k, v in UserResponseIndex.items()}[correctLabel]] = {
                'androidResult' : totalCorrectInCurrentRow,
                'result': totalInCurrentRow,
                'accuracy': totalCorrectInCurrentRow / totalInCurrentRow if totalInCurrentRow != 0 else 'N/A'
            }

        print('Total number of PCR Result data: ', lines)
        print('Number of correct prediction: ', totalCorrect)
        print('Percentage correct: ', totalCorrect /
              lines if lines != 0 else 'N/A')
        print('~~~~~~~~~~~~')
        currResult['overall'] = {
            'androidResult': totalCorrect,
            'result': lines,
            'accuracy': totalCorrect /
              lines if lines != 0 else 'N/A'
        }
        result.append({'User Response': currResult})
        self.printF1ScoreUserResponse(currResult)
        self.printResultTable(result)

    def printF1ScoreUserResponse(self, result):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
        f1Score, precision, recall = self.calculateF1ScoreUserResponse(result)
        print('F1 Score: ', f1Score)
        print('Precision: ', precision)
        print('Recall: ', recall)


    def calculateF1ScoreUserResponse(self, result):
        truePositive = result['Positive']['androidResult']
        falsePositive = result['Negative']['result'] - result['Negative']['androidResult']
        falseNegative = result['Positive']['result'] - result['Positive']['androidResult']
        trueNegative = result['Negative']['androidResult']
        print('[INFO] calculating f1 score', truePositive, falsePositive, falseNegative, trueNegative)
        precision = calculatePrecisionScore(truePositive, falsePositive)
        recall = calculateRecallScore(truePositive, falseNegative)
        f1Score = calculateF1Score(precision, recall)
        return (f1Score, precision, recall)

    def printResultTable(self, result):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~')
        lines = ["", "", "", "", "", "", "", ""]
        print(result)
        for category in result:
            lines[0] += str(list(category.keys())[0]) + "                               "
            categoryResult = category[list(category.keys())[0]]
            i = 1
            print(categoryResult)
            for k in categoryResult:
                v = categoryResult[k]
                # print(str(k) + " " + str(v['accuracy']) + "% " + "(" + str(v['androidResult']) + "/" + str(v['result']) + ")     ")
                lines[i] += str(k) + " " + str(v['accuracy'] * 100) + "% " + "(" + str(v['androidResult']) + "/" + str(v['result']) + ")                   "
                i += 1
                # print(lines[i])
        for line in lines:
            print(line)

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
            print('Total number of Labels: ', totalInCurrentRow)
            print('Percentage Accuracy: ',
                  totalCorrectInCurrentRow / totalInCurrentRow if totalInCurrentRow != 0 else 'N/A')
            print('~~~~~~~~')

        print('Total number of lines data: ', lines)
        print('Number of correct prediction: ', totalCorrect)
        print('Percentage correct: ', totalCorrect /
              lines if lines != 0 else 'N/A')
        print('~~~~~~~~~~~~')
        # print('Fail Detection Count', self.failDetectionCount)
        # print('Fail detection list')
        # print(json.dumps(self.failDetectionDetailList, sort_keys=True, indent=4))

    def printTable(self, rowLabels, colLabels, table):
        # if (len(table) > 1):
        #     s = "      "
        #     for colLabel in colLabels:
        #         print(colLabel),
        #         print('| ')

        # for i, row in enumerate(table):
        #     print(rowLabels[i]), 
        #     print('| ')
        #     for j, col in enumerate(row):
        #         print(col), 
        #         print(' ')

        for i, row in enumerate(table):
            print(row)

