import enum


class ExposureResult(enum.Enum):
    UNDER_EXPOSED = 0
    NORMAL = 1
    OVER_EXPOSED = 2


class CaptureResult():
    def __init__(self, allChecksPassed, imgResult, matchDistance, exposureResult, sizeResult,
                 isCentered, isRightOrientation, isSharp, isShadow, angle):
        self.allChecksPassed = allChecksPassed
        self.imgResult = imgResult
        self.matchDistance = matchDistance
        self.exposureResult = exposureResult
        self.sizeResult = sizeResult
        self.isCentered = isCentered
        self.isRightOrientation = isRightOrientation
        self.isSharp = isSharp
        self.isShadow = isShadow
        self.angle = angle

    # For call to repr(). Prints object's information
    def __repr__(self):
        return 'CaptureResult(%s, %s, %s, %s, %s, %s, %s, %s)' % (self.allChecksPassed, self.exposureResult, self.sizeResult,
                                                                  self.isCentered, self.isRightOrientation, self.isSharp, self.isShadow, self.angle)

    # For call to str(). Prints readable form
    def __str__(self):
        return 'CaptureResult(%s, %s, %s, %s, %s, %s, %s, %s)' % (self.allChecksPassed, self.exposureResult, self.sizeResult,
                                                                  self.isCentered, self.isRightOrientation, self.isSharp, self.isShadow, self.angle)


class InterpretationResult():
    def __init__(self, resultImg=None, control=None, testA=None, testB=None):
        self.resultImg = resultImg
        self.control = control
        self.testA = testA
        self.testB = testB

    def __str__(self):
        return 'InterpretResult(Control Line: %s, Test A: %s, TestB: %s)' % (self.control, self.testA, self.testB)

    def __repr__(self):
        return 'InterpretResult(Control Line: %s, Test A: %s, TestB: %s)' % (self.control, self.testA, self.testB)


class SizeResult(enum.Enum):
    RIGHT_SIZE = 0
    LARGE = 1
    SMALL = 2
    INVALID = 3
