from ImageProcessor import ImageProcessor

INPUT_IMAGE = 'input/testimg1.jpg'

imgProc = ImageProcessor(INPUT_IMAGE)
imgProc.captureRDT(INPUT_IMAGE)
# img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
# print(img.shape)
# cv2.imshow('Img', img)
# cv2.waitKey(0)

# # Check brightness
# # exposureResult = (checkBrightness(img))
# captureRDT(INPUT_IMAGE)
