# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:19:19 2019

"""

import sys 
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('C:\\Users\\Khushi\\Desktop\\ml\\bone\\boneage-training-dataset.csv')
df1=df[(df['id']>5660) & (df['id']<=8580)]
#imageList = ['F:/boneage-training-dataset/boneage-training-dataset/' + str(i) + '.png' for i in df1['id']]
imageList= glob.glob("F:\\boneage-training-dataset\\boneage-training-dataset\\3055.png")

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 400, 400)
for img_name in imageList:
    
    img = cv2.imread(img_name,0)
    mask = np.zeros_like(img)
    r = cv2.selectROI("Image",img)
    
    imCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    #rectangle=(r[0],r[1],r[2],r[3])
    mask = cv2.rectangle(mask,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),255,-1)
    #img = img * mask2[:, :, np.newaxis]
    im=np.where(mask==0,0,img)
    cv2.imwrite('F:\\boneage segmented data set\\'+img_name[-8:],im)
    cv2.imwrite('F:\\boneage segmented data set\\'+img_name[-8:],imCrop)
    print(img_name[-8:],'done')
    plt.imshow(im)