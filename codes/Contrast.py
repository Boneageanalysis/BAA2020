# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:18:54 2020

@author: Khushi
"""

import numpy as np
import argparse
import pandas as pd
import cv2
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)




df = pd.read_csv('H:\\boneage\\boneage-training-dataset.csv')
print(df.count())
imageList = ['H:\\boneage\\boneage-training-dataset\\' + str(i) + '.png' for i in df['id']]

print(len(imageList))
for img_name in imageList:
    print(img_name[-8:])

original=cv2.imread("C://Users//Khushi//Desktop//"+"1434"+".png")
'''
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
# load the original image
original = cv2.imread(args["image"])
'''
'''
adjusted = adjust_gamma(original, gamma=3)
cv2.imshow("Images", adjusted)
	  
'''
cv2.namedWindow('Images',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Images',500,500)

	# apply gamma correction and show the images
gamma = 1.5
adjusted = adjust_gamma(original, gamma=gamma)
cv2.putText(adjusted, "g={}".format(gamma), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)


cv2.imwrite('H:\\boneage\\gammaPY\\roii.png',adjusted)
cv2.imshow("Images", adjusted)
cv2.waitKey(0)
    




    
    
