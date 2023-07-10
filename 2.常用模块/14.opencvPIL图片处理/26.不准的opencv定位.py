import sys

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('data/cutcard/1.png')
mser = cv2.MSER_create(_min_area=300)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
regions, boxes = mser.detectRegions(gray)

for box in boxes:
    x, y, w, h = box
    cv2.rectangle(img, (x,y),(x+w, y+h), (255, 0, 0), 2)

plt.imshow(img,'brg')
plt.show()