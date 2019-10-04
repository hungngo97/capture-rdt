import numpy as np
import json

variables = {}
with open('variables/variables.json') as json_file:
    variables = json.load(json_file)
# Brightness
OVER_EXP_THRESHOLD = 255
UNDER_EXP_THRESHOLD = 120
OVER_EXP_WHITE_COUNT = 100

# Sharpness
VIEW_FINDER_SCALE_H = 0.60
VIEW_FINDER_SCALE_W = 0.15
SHARPNESS_THRESHOLD = 0.7

CROP_RATIO = 1.0

# Intensity
INTENSITY_THRESHOLD = 190
CONTROL_INTENSITY_PEAK_THRESHOLD = 150
TEST_INTENSITY_PEAK_THRESHOLD = 50

# Homography
MIN_MATCH_COUNT = 7

# Is centered
POSITION_THRESHOLD = 0.15
SIZE_THRESHOLD = 0.20

# Oritentation
ANGLE_THRESHOLD = 10

#
CONTROL_LINE_POSITION = 45  # Old value45
TEST_A_LINE_POSITION = 15
TEST_B_LINE_POSITION = 75  # Old value 75
DETECTION_RANGE = 14

#
CONTROL_LINE_COLOR_UPPER = np.array(
    [60 / 1.0, 90/100.0*255.0, 100/100.0*255.0])
CONTROL_LINE_COLOR_LOWER = np.array([5 / 1.0, 20/100.0*255.0, 5/100.0*255.0])
LINE_SEARCH_WIDTH = 13

CONTROL_LINE_POSITION_MIN = 575 * 2
CONTROL_LINE_POSITION_MAX = 700 * 2
CONTROL_LINE_MIN_HEIGHT = 25
CONTROL_LINE_MIN_WIDTH = 20
CONTROL_LINE_MAX_WIDTH = 55

# Fiducial
FIDUCIAL_POSITION_MIN = 160
FIDUCIAL_POSITION_MAX = 935
FIDUCIAL_MIN_HEIGHT = 30  # this should be 30
FIDUCIAL_MIN_WIDTH = 20
FIDUCIAL_MAX_WIDTH = 160
FIDUCIAL_TO_CONTROL_LINE_OFFSET = 50
RESULT_WINDOW_RECT_HEIGHT = 90
RESULT_WINDOW_RECT_WIDTH_PADDING = 20
ANGLE_THRESHOLD = 10
FIDUCIAL_DISTANCE = 610
FIDUCIAL_COUNT = 2

# COLOR DETECTION
NUM_OF_NEIGHBOR_COLOR_DETECTION = 10

# Constants from variables json file
PEAK_HEIGHT_THRESHOLD = variables["PEAK_HEIGHT_THRESHOLD"]
