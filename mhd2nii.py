import SimpleITK as sitk
import os
import cv2
import numpy as np

filelist = []
for root, dirs, files in os.walk("dataset/training"):
    for file in files:
        if file.endswith(".mhd"):
            filelist.append(os.path.join(root, file))

shapes = []
for i in filelist:
    img = sitk.ReadImage(i)
    vol = sitk.GetArrayFromImage(img)
    vol = np.transpose(vol,[0,1,2])
    mip = vol.max(axis=0)
    shapes.append(vol.shape)
    imgshow = (mip/mip.max()*255).astype(np.uint8)
    cv2.imshow('mip',(mip/mip.max()*255).astype(np.uint8))
    cv2.imwrite('data/train/%spng'%os.path.split(i)[1][:-3] , imgshow)
    # cv2.waitKey(0)
    sitk.WriteImage(img,"data/train/"+os.path.split(i)[1][:-3]+"nii.gz")
# print(shapes)
# print(np.array(shapes).min(0))
