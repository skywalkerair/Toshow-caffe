import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("C:/Users/Administrator/Desktop/Caffe_Using/caffe-master/myself/classification_test/501.jpg")
(r,g,b)=cv2.split(img)

img = cv2.merge([b,g,r])

plt.imshow(img)
plt.show()