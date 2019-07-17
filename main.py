from ImageProcessor import ImageProcessor

INPUT_IMAGE = 'input/testimg8_1.jpg'
# 4.1 doesnot work well because the arrow got cut in half


imgProc = ImageProcessor(INPUT_IMAGE)
# imgProc.captureRDT(INPUT_IMAGE)
imgProc.interpretResult(INPUT_IMAGE)
