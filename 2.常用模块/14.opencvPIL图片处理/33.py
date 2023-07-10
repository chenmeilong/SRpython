import cv2
import numpy as np

imgFix = np.zeros((720,1280, 3), np.uint8)

cv2.imshow("imgFix", imgFix)
cv2.waitKey()