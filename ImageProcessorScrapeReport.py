from __future__ import division
from ImageProcessorScrape import ImageProcessorScrape
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from urlConstants import (
    S3_URL_BASE_PATH, TYPE, BARCODES, RDT_SCAN, ENHANCED_SCAN, MANUAL_PHOTO
)
from utils import (
    calculateF1Score, calculatePrecisionScore, calculateRecallScore,
    calculateFalsePositiveRate, calculateTruePositiveRate, calculateROCStats
)
import math
import json
import sys
"""
    TODO: 
     ** Refactor reportAndroidResultStatistics and reportPythonResultStatistics (NOT DONE)
     ** Refactor androidResult where it should be pythonResult

"""

SECRET_PATH = 'keys/cough_photos_key.txt'
STATUS = 'Status'
PCR_RESULT = 'ASPREN: PCR Result'
RESULTS_USER_RESPONSE = 'Results: Shown to User Based Only on User Responses'
RDT_RESULT = 'RDT Result: What the RDT Algorithm Interpreted'
HIGH_CONTRAST_LINE_ANSWER = 'High Contrast Line Answer'
TEST_STRIP_BOUNDARY = 'Test Strip Boundary'
BARCODE = 'Barcode'
EXPERT_RESPONSE = 'Strip Line Answer (expert)'
SECRET = ''
ANDROID = 'ANDROID'
PYTHON = 'PYTHON'
with open(SECRET_PATH, 'r') as file:
    SECRET = file.read()

IntepretationResultMappings = {
    'Both': 0,
    'Negative': 0,
    'Flu B': 0,
    'Flu A': 0,
    'No control line': 0
}

ExpertResponseMappings = {
    'noPink': 0,
    'yesAboveBlue': 0,
    'badPicture': 0,
    'yesBelowBlue': 0,
    'yesAboveAndBelowBlue': 0,
    'noBlue': 0
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

ExpertResponseIndex = {
    'noPink': 0,
    'yesAboveBlue': 1,
    'badPicture': 2,
    'yesBelowBlue': 3,
    'yesAboveAndBelowBlue': 4,
    'noBlue': 5
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
    def __init__(self, output_path):
        self.colLabels = ['No interpretation',
                          'Both', 'Test A', 'Test B', 'No Flu']
        ImageProcessorScrape.__init__(self, output_path)
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
            [0, 0, 0, 0, 0, 0] for _ in range(len(HighContrastLineMappings))]
        self.resultPythonComparisonWithUserResponse = [
            [0, 0, 0, 0, 0, 0] for _ in range(len(UserResponseMappings))]
        self.resultPythonComparisonWithPCRResult = [
            [0, 0, 0, 0, 0, 0] for _ in range(len(PCRMappings))]
        self.resultPythonComparisonWithExpertResponse = [
            [0, 0, 0, 0, 0, 0] for _ in range(len(ExpertResponseMappings))]
        self.resultAndroidComparisonWithHighContrastLineAnswer = [
            [0, 0, 0, 0, 0, 0] for _ in range(len(HighContrastLineMappings))]
        self.resultAndroidComparisonWithUserResponse = [
            [0, 0, 0, 0, 0, 0] for _ in range(len(UserResponseMappings))]
        self.resultAndroidComparisonWithPCRResult = [
            [0, 0, 0, 0, 0, 0] for _ in range(len(PCRMappings))]
        self.resultAndroidComparisonWithExpertResponse = [
            [0, 0, 0, 0, 0, 0] for _ in range(len(ExpertResponseMappings))]
        self.lineCountResult = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(4)]
        self.failDetectionCount = 0
        self.failDetectionDetailList = []

    def preprocessTestStripBoundary(self, test_strip_boundary):
        print('[INFO] teststripboundary', test_strip_boundary)
        boundary = json.loads(test_strip_boundary)
        print('[INFO] Test Strip Boundary JSON', test_strip_boundary)
        print('[INFO] Test strip map conversion', boundary)
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
                       rdt_result, high_contrast_line_answer, test_strip_boundary, expert_response):
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
        interpretResult = self.interpretResultFromURL(
            URL_PATH, URL_PATH, boundary)
        if interpretResult is None:
            # ===== TODO: Do something else if interpretResult is None
            self.failDetectionCount += 1
            failDetectionDetail = {
                "barcode": barcode,
                "url": URL_PATH,
                "pcrResult": pcr_result,
                "resultsUserResponse": results_user_response,
                "rdtResult": rdt_result,
                "expertResponse": expert_response,
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
        if expert_response and (isinstance(expert_response, str) or not math.isnan(expert_response)):
            self.compareExpertResponse(interpretResult, expert_response)
        self.compareAndroidResult(
            rdt_result, pcr_result, results_user_response, high_contrast_line_answer, expert_response)

    """
        Output: To generate the truth matrix for each category
    """
    def compareAndroidResult(self, rdt_result, pcr_result, results_user_response, high_contrast_line_answer, expert_response):
        print('[INFO] start compareAndroidResult')
        if not (high_contrast_line_answer and (isinstance(high_contrast_line_answer, str) or not math.isnan(high_contrast_line_answer))):
            return None
        rdtResultColumnIndex = IntepretationResultMappingsIndex[rdt_result]

        if (high_contrast_line_answer and (isinstance(high_contrast_line_answer, str) or not math.isnan(high_contrast_line_answer))):
            contrastLineRowIndex = HighContrastLineIndex[high_contrast_line_answer]
            self.resultAndroidComparisonWithHighContrastLineAnswer[
                contrastLineRowIndex][rdtResultColumnIndex] += 1

        if (expert_response and (isinstance(expert_response, str) or not math.isnan(expert_response))):
            expertResponseRowIndex = ExpertResponseIndex[expert_response]
            self.resultAndroidComparisonWithExpertResponse[
                expertResponseRowIndex][rdtResultColumnIndex] += 1

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

    def calculateColumnIndexForInterpretResult(self, interpretResult):
        col_index = 0
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
            col_index += 5
        return col_index

    def comparePCRResult(self, interpretResult, pcr_result):
        print('[INFO] start comparePCRResult')
        row_index = PCRMappingsIndex[pcr_result]
        col_index = self.calculateColumnIndexForInterpretResult(interpretResult)
        print('[INFO] row , column indices: ', row_index, col_index)
        self.resultPythonComparisonWithPCRResult[row_index][col_index] += 1

    def compareUserResponse(self, interpretResult, results_user_response):
        print('[INFO] start comparePCRResult')
        row_index = UserResponseIndex[results_user_response]
        col_index = self.calculateColumnIndexForInterpretResult(interpretResult)
        print('[INFO] row , column indices: ', row_index, col_index)
        self.resultPythonComparisonWithUserResponse[row_index][col_index] += 1

    def compareHighContrastLine(self, interpretResult, high_contrast_line_answer):
        print('[INFO] start comparePCRResult')
        row_index = HighContrastLineIndex[high_contrast_line_answer]
        col_index = self.calculateColumnIndexForInterpretResult(interpretResult)
        print('[INFO] row , column indices: ', row_index, col_index)
        self.resultPythonComparisonWithHighContrastLineAnswer[row_index][col_index] += 1

    def compareExpertResponse(self, interpretResult, expert_response):
        print('[INFO] start compareExpertResponse')
        row_index = ExpertResponseIndex[expert_response]
        col_index = self.calculateColumnIndexForInterpretResult(interpretResult)
        print('[INFO] row , column indices: ', row_index, col_index)
        self.resultPythonComparisonWithExpertResponse[row_index][col_index] += 1

    def printCSVFileRow(self, row):
        print("========= [ROW INFO] ===========")
        print(BARCODE, row[BARCODE])
        print(PCR_RESULT, row[PCR_RESULT])
        print(RESULTS_USER_RESPONSE, row[RESULTS_USER_RESPONSE])
        print(RDT_RESULT, row[RDT_RESULT])
        print(HIGH_CONTRAST_LINE_ANSWER,
                row[HIGH_CONTRAST_LINE_ANSWER])
        print(TEST_STRIP_BOUNDARY, row[TEST_STRIP_BOUNDARY])
        print(EXPERT_RESPONSE, row[EXPERT_RESPONSE])
        print(type(row[RDT_RESULT]))
        print("=============================")

    def incrementCategoryMapping(self, row):
        if row[RDT_RESULT] and (isinstance(row[RDT_RESULT], str) or not math.isnan(row[RDT_RESULT])):
            IntepretationResultMappings[row[RDT_RESULT]] += 1
        if row[PCR_RESULT] and isinstance(row[PCR_RESULT], str) or not math.isnan(row[PCR_RESULT]):
            PCRMappings[row[PCR_RESULT]] += 1
        if row[RESULTS_USER_RESPONSE]and isinstance(row[RESULTS_USER_RESPONSE], str) or not math.isnan(row[RESULTS_USER_RESPONSE]):
            UserResponseMappings[row[RESULTS_USER_RESPONSE]] += 1
        if row[HIGH_CONTRAST_LINE_ANSWER] and isinstance(row[HIGH_CONTRAST_LINE_ANSWER], str) or not math.isnan(row[HIGH_CONTRAST_LINE_ANSWER]):
            HighContrastLineMappings[row[HIGH_CONTRAST_LINE_ANSWER]] += 1
        if row[EXPERT_RESPONSE] and isinstance(row[EXPERT_RESPONSE], str) or not math.isnan(row[EXPERT_RESPONSE]):
            ExpertResponseMappings[row[EXPERT_RESPONSE]] += 1

    def printReport(self, total, validBarcodes):
        print('+++++++++++++++++++++++++++++++REPORT+++++++++++++++++++++++++++++++++++++')
        print('----------------------------Overall Statistics----------------------------')
        self.printOverallStatistics(total, validBarcodes)
        print('--------------------------Accuracy Table Comparison-----------------------')
        print('=======Python Result========')
        self.printPlatformCategoryMatrix(PYTHON)
        pythonResultStats = self.reportPythonResultStatistics()
        print('=======Android Result=========')
        self.printPlatformCategoryMatrix(ANDROID)
        self.reportAndroidResultStatistics()

        self.printLineCountStatistics()
        # self.printFailCasesStatistics()

        return pythonResultStats

    def printOverallStatistics(self, total, validBarcodes):
        print('Total data rows: ', total)
        print('Valid barcodes: ', validBarcodes)
        print('High contrast Mappings', HighContrastLineMappings)
        print('Interpretation Result Mappings', IntepretationResultMappings)
        print('User Response Mappings', UserResponseMappings)
        print('Expert Response Mappings', ExpertResponseMappings)

    def printPlatformCategoryMatrix(self, result_platform):
        if result_platform == ANDROID:
            print('High Contrast Line Result Table')
            self.printTable(HighContrastLineIndex.keys, self.colLabels,
                            self.resultAndroidComparisonWithHighContrastLineAnswer)
            print('PCR Result Table')
            self.printTable(PCRMappings.keys, self.colLabels,
                            self.resultAndroidComparisonWithPCRResult)
            print('User Response Result Table')
            self.printTable(HighContrastLineIndex.keys, self.colLabels,
                            self.resultAndroidComparisonWithUserResponse)
            print('Expert Response Mappings')
            self.printTable(ExpertResponseIndex.keys, self.colLabels,
                            self.resultAndroidComparisonWithExpertResponse)
        elif result_platform == PYTHON:
            print('High Contrast Line Result Table')
            self.printTable(HighContrastLineIndex.keys, self.colLabels,
                            self.resultPythonComparisonWithHighContrastLineAnswer)
            print('PCR Result Table')
            self.printTable(PCRMappings.keys, self.colLabels,
                            self.resultPythonComparisonWithPCRResult)
            print('User Response Result Table')
            self.printTable(UserResponseMappings.keys, self.colLabels,
                            self.resultPythonComparisonWithUserResponse)
            print('Expert Response Mappings')
            self.printTable(ExpertResponseMappings.keys, self.colLabels,
                            self.resultPythonComparisonWithExpertResponse)

            

    def printLineCountStatistics(self):
        print('=======Line Count=======')
        print('Line count table')
        self.printTable(range(4), range(5), self.lineCountResult)
        print('~~~~~~~')
        self.reportLineCountStatistics()

    def printFailCasesStatistics(self):
        print('============FAIL==========')
        print('Fail Detection Count', self.failDetectionCount)
        print('Fail detection list')
        print(json.dumps(self.failDetectionDetailList, sort_keys=True, indent=4))

    def processFile(self, file, db):
        print('[INFO] processing filename ', file)
        df = pd.read_csv(file)
        print(df.head(5))
        print("[INFO] columns")
        print(df.columns)
        validBarcodes = 0
        total = 0
        barcodes = []
        # ================ DEBUG =========================
        DEBUG_AMOUNT = int(db) if db else None
        DEBUG_counter = 1

        for index, row in df.iterrows():
            # try:
            if row[BARCODE]:
                # REPORT
                self.printCSVFileRow(row)
                validBarcodes += 1
                self.incrementCategoryMapping(row)
                self.processBarcode(row[BARCODE], row[PCR_RESULT], row[RESULTS_USER_RESPONSE],
                                    row[RDT_RESULT], row[HIGH_CONTRAST_LINE_ANSWER], row[TEST_STRIP_BOUNDARY], row[EXPERT_RESPONSE])

                # BREAK DEBUG
                DEBUG_counter += 1
                if (db and DEBUG_counter > DEBUG_AMOUNT):
                    break
            total += 1
            # except:
            #     continue

        pythonResultStats = self.printReport(total, validBarcodes)
        return pythonResultStats 
        #For further analysis like in roc curve to compare results to 
        #optimize hyperparameter

    def matchCategoryVariables(self, category, result_platform):
        datatable, title, mapping_index = None, None, None
        if result_platform == PYTHON:
            if (category == HIGH_CONTRAST_LINE_ANSWER):
                datatable = self.resultPythonComparisonWithHighContrastLineAnswer if result_platform == PYTHON else self.resultAndroidComparisonWithHighContrastLineAnswer
                title = HIGH_CONTRAST_LINE_ANSWER
                mapping_index = HighContrastLineIndex
            elif category == PCR_RESULT:
                datatable = self.resultPythonComparisonWithPCRResult
                title = PCR_RESULT
                mapping_index = PCRMappingsIndex
            elif category == EXPERT_RESPONSE:
                datatable = self.resultPythonComparisonWithExpertResponse
                title = EXPERT_RESPONSE
                mapping_index = ExpertResponseIndex
            elif category == RESULTS_USER_RESPONSE:
                datatable = self.resultPythonComparisonWithUserResponse
                title = RESULTS_USER_RESPONSE
                mapping_index = UserResponseIndex
        elif result_platform == ANDROID:
            if (category == HIGH_CONTRAST_LINE_ANSWER):
                datatable = self.resultAndroidComparisonWithHighContrastLineAnswer
                title = HIGH_CONTRAST_LINE_ANSWER
                mapping_index = HighContrastLineIndex
            elif category == PCR_RESULT:
                datatable = self.resultAndroidComparisonWithPCRResult
                title = PCR_RESULT
                mapping_index = PCRMappingsIndex
            elif category == EXPERT_RESPONSE:
                datatable = self.resultAndroidComparisonWithExpertResponse
                title = EXPERT_RESPONSE
                mapping_index = ExpertResponseIndex
            elif category == RESULTS_USER_RESPONSE:
                datatable = self.resultAndroidComparisonWithUserResponse
                title = RESULTS_USER_RESPONSE
                mapping_index = UserResponseIndex
        return (datatable, title, mapping_index)


    def isPredictionCorrect(self, correctLabel, i, category, result_platform):
        """
            TODO: finish this method
        """
        if result_platform == PYTHON:
            if (category == HIGH_CONTRAST_LINE_ANSWER):
                # Column
                # No interpretation = 0
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
                return (correctLabel == 0 and i == 0) or \
                    ((correctLabel == 1 and i == 4)) or \
                    (correctLabel == 2 and (i == 2 or i == 3)) or \
                        (correctLabel == 3 and i == 1)
            elif category == PCR_RESULT:
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
                return (correctLabel == 0 and (i == 4 or i == 0)) or \
                    ((correctLabel == 1 and (i == 2 or i == 1))) or \
                        (correctLabel == 2 and (i == 1 or i == 3))
            elif category == EXPERT_RESPONSE:
                # Expert Reponse
                # Column
                # No interpretation = 0
                # Both = 1
                # testA = 2
                # testB = 3
                # No flu = 4
                # Row
                #  'noPink': 0,
                # 'yesAboveBlue': 1,
                # 'badPicture': 2,
                # 'yesBelowBlue': 3,
                # 'yesAboveAndBelowBlue': 4,
                # 'noBlue': 5
                return (correctLabel == 0 and (i == 4)) or \
                        ((correctLabel == 1 and (i == 2))) or \
                        (correctLabel == 3 and i == 3) or \
                        (correctLabel == 4 and i == 1) or \
                        (correctLabel == 5 and i == 0)
            elif category == RESULTS_USER_RESPONSE:
                # User Reponse
                # 'Negative': 0,
                # 'Positive': 1
                return (correctLabel == 0 and (i == 4)) or \
                        ((correctLabel == 1 and (i != 0 and i != 4)))
        elif result_platform == ANDROID:
            if (category == HIGH_CONTRAST_LINE_ANSWER):
                # Column
                # 'Both': 0,
                # 'Negative': 1,
                # 'Flu B': 2,
                # 'Flu A': 3,
                # 'No control line': 4
                # ROw
                #  'oneLine': 1,
                # 'noneOfTheAbove': 0,
                # 'twoLines': 2,
                # 'threeLines': 3
                return (correctLabel == 0 and i == 1) or \
                    ((correctLabel == 1 and i == 4)) or \
                    (correctLabel == 2 and (i == 2 or i == 3)) or \
                        (correctLabel == 3 and i == 0)
            elif category == PCR_RESULT:
                # Column
                # 'Both': 0,
                # 'Negative': 1,
                # 'Flu B': 2,
                # 'Flu A': 3,
                # 'No control line': 4
                # PCR Result
                #   'negative': 0,
                # 'flu A': 1,
                # 'flu B': 2
                return (correctLabel == 0 and (i == 1)) or \
                    ((correctLabel == 1 and (i == 3))) or \
                        (correctLabel == 2 and (i == 2))
            elif category == EXPERT_RESPONSE:
                # Expert Reponse
                # Column
                # 'Both': 0,
                # 'Negative': 1,
                # 'Flu B': 2,
                # 'Flu A': 3,
                # 'No control line': 4
                # Row
                #  'noPink': 0,
                # 'yesAboveBlue': 1,
                # 'badPicture': 2,
                # 'yesBelowBlue': 3,
                # 'yesAboveAndBelowBlue': 4,
                # 'noBlue': 5
                return (correctLabel == 0 and (i == 1)) or \
                        ((correctLabel == 1 and (i == 3))) or \
                        (correctLabel == 3 and i == 2) or \
                        (correctLabel == 4 and i == 0) or \
                        (correctLabel == 5 and i == 4)
            elif category == RESULTS_USER_RESPONSE:
                # User Reponse
                # 'Both': 0,
                # 'Negative': 1,
                # 'Flu B': 2,
                # 'Flu A': 3,
                # 'No control line': 4
                # Column
                # 'Negative': 0,
                # 'Positive': 1
                return (correctLabel == 0 and (i == 1)) or \
                        ((correctLabel == 1 and (i != 1)))
        return None

    def generateCategoryResultStatistics(self, category, result_platform):
        if result_platform != PYTHON and result_platform != ANDROID:
            raise Exception('Invalid Platform', result_platform)

        totalCorrect = 0
        lines = 0
        currResult = {}
        datatable, title, mappingIndex = self.matchCategoryVariables(category, result_platform)


        for correctLabel, row in enumerate(datatable):
            totalCorrectInCurrentRow = 0
            totalInCurrentRow = 0
            for i, num in enumerate(row):
                if self.isPredictionCorrect(correctLabel, i, category, result_platform):
                    totalCorrect += num
                    totalCorrectInCurrentRow += num
                lines += num
                totalInCurrentRow += num
            print('True Label' + str(title) + ": ",
                  {v: k for k, v in mappingIndex.items()}[correctLabel])
            print('Number of correct prediction: ', totalCorrectInCurrentRow)
            print('Total number of Labels: ', totalInCurrentRow)
            print('Percentage Accuracy: ',
                  totalCorrectInCurrentRow / totalInCurrentRow if totalInCurrentRow != 0 else 'N/A')
            print('~~~~~~~~')
            currResult[{v: k for k, v in mappingIndex.items()}[correctLabel]] = {
                'correctPrediction': totalCorrectInCurrentRow,
                'totalLabels': totalInCurrentRow,
                'accuracy': totalCorrectInCurrentRow / totalInCurrentRow if totalInCurrentRow != 0 else 'N/A'
            }

        print('Total number of ' + str(title) + ' category: ', lines)
        print('Number of correct prediction: ', totalCorrect)
        print('Percentage correct: ', totalCorrect /
              lines if lines != 0 else 'N/A')
        print('~~~~~~~~~~~~')
        currResult['overall'] = {
            'correctPrediction': totalCorrect,
            'totalLabels': lines,
            'accuracy': totalCorrect /
            lines if lines != 0 else 'N/A'
        }
        return currResult

    def reportPythonResultStatistics(self):
        result = []
        highContrastLineResultStats = self.generateCategoryResultStatistics(HIGH_CONTRAST_LINE_ANSWER, PYTHON)
        pcrResultStats = self.generateCategoryResultStatistics(PCR_RESULT, PYTHON)
        userResponseStats = self.generateCategoryResultStatistics(RESULTS_USER_RESPONSE, PYTHON)
        expertResponseStats = self.generateCategoryResultStatistics(EXPERT_RESPONSE, PYTHON)
        
        result.append({'High Contrast': highContrastLineResultStats})
        result.append({'PCR': pcrResultStats})

        result.append({'User Response': userResponseStats})
        self.printF1ScoreUserResponse(userResponseStats)

        result.append({'Expert Response': expertResponseStats})
        python_expert_response = self.printF1ScoreExpertResponse(
            expertResponseStats, PYTHON)
        self.printResultTable(result)
        
        # Fix the return value here if wanted to compare different results
        return python_expert_response

    def reportAndroidResultStatistics(self):
        result = []
        highContrastLineResultStats = self.generateCategoryResultStatistics(HIGH_CONTRAST_LINE_ANSWER, ANDROID)
        pcrResultStats = self.generateCategoryResultStatistics(PCR_RESULT, ANDROID)
        userResponseStats = self.generateCategoryResultStatistics(RESULTS_USER_RESPONSE, ANDROID)
        expertResponseStats = self.generateCategoryResultStatistics(EXPERT_RESPONSE, ANDROID)
        
        result.append({'High Contrast': highContrastLineResultStats})
        result.append({'PCR': pcrResultStats})

        result.append({'User Response': userResponseStats})
        self.printF1ScoreUserResponse(userResponseStats)

        result.append({'Expert Response': expertResponseStats})
        android_expert_response = self.printF1ScoreExpertResponse(
            expertResponseStats, ANDROID)
        self.printResultTable(result)
        
        # Fix the return value here if wanted to compare different results
        return android_expert_response

    def findTruthMatrixValues(self, result_table, category):
        if category == EXPERT_RESPONSE:
            # Expert Reponse
            # Column
            # No interpretation = 0
            # Both = 1
            # testA = 2
            # testB = 3
            # No flu = 4
            # Row
            #  'noPink': 0,
            # 'yesAboveBlue': 1,
            # 'badPicture': 2,
            # 'yesBelowBlue': 3,
            # 'yesAboveAndBelowBlue': 4,
            # 'noBlue': 5
            falseNegative, falsePositive, trueNegative, truePositive = 0, 0, 0, 0
            for correctLabel, row in enumerate(result_table):
                totalCorrectInCurrentRow = 0
                totalInCurrentRow = 0
                for i, num in enumerate(row):
                    if correctLabel == 0:  # No Pink: Negative
                        if i == 4:
                            trueNegative += num
                        elif i == 1 or i == 2 or i == 3:
                            falsePositive += num
                    elif correctLabel == 1 or correctLabel == 3 or correctLabel == 4:
                        if (correctLabel == 1 and i == 2) or \
                            (correctLabel == 3 and i == 3) or \
                                (correctLabel == 4 and i == 1):  # Positive cases
                            truePositive += num
                        elif i == 4:
                            falseNegative += num
                    # Ignore noBlue and bad Picture
            print('[INFO] calculating f1 score', truePositive,
                falsePositive, falseNegative, trueNegative)
            return (falseNegative, falsePositive, trueNegative, truePositive)

    def printF1ScoreExpertResponse(self, result, result_platform):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
        result_table = None
        if result_platform == ANDROID:
            result_table = self.resultAndroidComparisonWithExpertResponse
        elif result_platform == PYTHON:
            result_table = self.resultPythonComparisonWithExpertResponse

        falseNegative, falsePositive, trueNegative, truePositive = self.findTruthMatrixValues(result_table, EXPERT_RESPONSE)
        f1Score, precision, recall, falsePositiveRate, truePositiveRate = calculateROCStats(truePositive,falsePositive, trueNegative, falseNegative)
        return {
            'python_expert_response': {
                'truePositive': truePositive,
                'falsePositive': falsePositive,
                'trueNegative': trueNegative,
                'falseNegative': falseNegative,
                'precision': precision,
                'recall': recall,
                'f1Score': f1Score,
                'falsePositiveRate': falsePositiveRate,
                'truePositiveRate': truePositiveRate,
                'confusionMatrix': result_table
            }
        }

    def printF1ScoreUserResponse(self, result):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('User Reponse')
        f1Score, precision, recall = self.calculateF1ScoreUserResponse(result)
        print('F1 Score: ', f1Score)
        print('Precision: ', precision)
        print('Recall: ', recall)

    def calculateF1ScoreUserResponse(self, result):
        truePositive = result['Positive']['correctPrediction']
        falsePositive = result['Negative']['totalLabels'] - \
            result['Negative']['correctPrediction']
        falseNegative = result['Positive']['totalLabels'] - \
            result['Positive']['correctPrediction']
        trueNegative = result['Negative']['correctPrediction']
        print('[INFO] calculating f1 score', truePositive,
              falsePositive, falseNegative, trueNegative)
        precision = calculatePrecisionScore(truePositive, falsePositive)
        recall = calculateRecallScore(truePositive, falseNegative)
        f1Score = calculateF1Score(precision, recall)
        return (f1Score, precision, recall)

    def printResultTable(self, result):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~')
        lines = ["", "", "", "", "", "", "", ""]
        print(result)
        for category in result:
            lines[0] += str(list(category.keys())[0]) + \
                "                               "
            categoryResult = category[list(category.keys())[0]]
            i = 1
            print(categoryResult)
            for k in categoryResult:
                v = categoryResult[k]
                if (v['accuracy'] != 'N/A'):
                    lines[i] += str(k) + " " + str(v['accuracy'] * 100) + "% " + "(" + str(
                        v['correctPrediction']) + "/" + str(v['totalLabels']) + ")                   "
                else: #To avoid N/A cases print 100 times
                    lines[i] += str(k) + " " + str(v['accuracy']) + "% " + "(" + str(
                        v['correctPrediction']) + "/" + str(v['totalLabels']) + ")                   "
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

    def printTable(self, rowLabels, colLabels, table):
        for i, row in enumerate(table):
            print(row)
