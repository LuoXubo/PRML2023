"""
@Description :   
@Author      :   Xubo Luo 
@Time        :   2024/01/07 21:51:51
"""

from skimage.feature import hog
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('../figs/cat.jpg')

RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.subplot(1, 2, 1)
plt.imshow(RGB)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ft, hogimg = hog(gray, 
                 orientations=9, 
                 pixels_per_cell=(8, 8), 
                 cells_per_block=(8, 8), 
                 visualize=True, 
                 block_norm='L2-Hys', 
                 transform_sqrt=True, 
                 feature_vector=True)
plt.subplot(1, 2, 2)
plt.imshow(hogimg)
plt.show()