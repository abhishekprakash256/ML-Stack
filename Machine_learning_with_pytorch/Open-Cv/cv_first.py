"""
using the opencv to read and fetch images
"""

import cv2
import numpy as np


# read image
img = cv2.imread(r"./images/pigeon.jpg", 1)

print(img.shape)

print(type(img))