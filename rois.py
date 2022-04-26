import SimpleITK as sitk
import numpy as np
from nii import savenii,nii2array

# mask,spacing,origin = nii2array("train/%s/input.nii.gz"%(29))

def cropROI(volume,outputsize,step):
    ROIs = []
    idxs = []
    num_x = (volume.shape[0]-outputsize[0]-1)//step+1
    num_y = (volume.shape[1]-outputsize[1]-1)//step+1
    num_z = (volume.shape[2]-outputsize[2]-1)//step+1
    print(num_x,num_y,num_z)
    for i in range(num_z):
        for j in range(num_y):
            for k in range(num_x):
                ROIs.append(volume[k*step:k*step+outputsize[0],
                            j*step:j*step+outputsize[1],
                            i*step:i*step+outputsize[2]])
                idxs.append([k*step,j*step,i*step])
            ROIs.append(volume[volume.shape[0]-outputsize[0]:,
                        j*step:j*step+outputsize[1],
                        i*step:i*step+outputsize[2]])
            idxs.append([volume.shape[0]-outputsize[0],j*step,i*step])
        for k in range(num_x):
            ROIs.append(volume[k*step:k*step+outputsize[0],
                                volume.shape[1]-outputsize[1]:,
                                i*step:i*step+outputsize[2]])
            idxs.append([k*step,volume.shape[1]-outputsize[1],i*step])
        ROIs.append(volume[volume.shape[0]-outputsize[0]:,
                            volume.shape[1]-outputsize[1]:,
                            i*step:i*step+outputsize[2]])
        idxs.append([volume.shape[0]-outputsize[0],volume.shape[1]-outputsize[1],i*step])
    print(len(ROIs))
    for j in range(num_y):
        for k in range(num_x):
            ROIs.append(volume[k*step:k*step+outputsize[0],
                                j*step:j*step+outputsize[1],
                                volume.shape[2]-outputsize[2]:])
            idxs.append([k*step,j*step,volume.shape[2]-outputsize[2]])
        ROIs.append(volume[volume.shape[0]-outputsize[0]:,
                            j*step:j*step+outputsize[1],
                            volume.shape[2]-outputsize[2]:])
        idxs.append([volume.shape[0]-outputsize[0],j*step,volume.shape[2]-outputsize[2]])
    for k in range(num_x):
        ROIs.append(volume[k*step:k*step+outputsize[0],
                    volume.shape[1]-outputsize[1]:,
                    volume.shape[2]-outputsize[2]:])
        idxs.append([k*step,volume.shape[1]-outputsize[1],volume.shape[2]-outputsize[2]])
    ROIs.append(volume[volume.shape[0]-outputsize[0]:,
                        volume.shape[1]-outputsize[1]:,
                        volume.shape[2]-outputsize[2]:])
    idxs.append([volume.shape[0]-outputsize[0],
                        volume.shape[1]-outputsize[1],
                        volume.shape[2]-outputsize[2]])
    return ROIs,idxs
#将截取的ROI恢复回去

def eraseROI(ROIs,inputsize,step):
    volume = np.zeros(inputsize)
    outputsize = ROIs[0].shape
    num_x = (inputsize[0]-outputsize[0]-1)//step+1
    num_y = (inputsize[1]-outputsize[1]-1)//step+1
    num_z = (inputsize[2]-outputsize[2]-1)//step+1
    for i in range(num_z):
        for j in range(num_y):
            for k in range(num_x):
                volume[k*step:k*step+outputsize[0],
                            j*step:j*step+outputsize[1],
                            i*step:i*step+outputsize[2]] = np.maximum.reduce([volume[k*step:k*step+outputsize[0],
                            j*step:j*step+outputsize[1],
                            i*step:i*step+outputsize[2]],ROIs.pop(0)])
            volume[volume.shape[0]-outputsize[0]:,
                        j*step:j*step+outputsize[1],
                        i*step:i*step+outputsize[2]] = np.maximum.reduce([volume[volume.shape[0]-outputsize[0]:,
                        j*step:j*step+outputsize[1],
                        i*step:i*step+outputsize[2]],ROIs.pop(0)])
        for k in range(num_x):
            volume[k*step:k*step+outputsize[0],
                                volume.shape[1]-outputsize[1]:,
                                i*step:i*step+outputsize[2]] = np.maximum.reduce([volume[k*step:k*step+outputsize[0],
                                volume.shape[1]-outputsize[1]:,
                                i*step:i*step+outputsize[2]],ROIs.pop(0)])
        volume[volume.shape[0]-outputsize[0]:,
                            volume.shape[1]-outputsize[1]:,
                            i*step:i*step+outputsize[2]] = np.maximum.reduce([volume[volume.shape[0]-outputsize[0]:,
                            volume.shape[1]-outputsize[1]:,
                            i*step:i*step+outputsize[2]],ROIs.pop(0)])
    print(len(ROIs))
    for j in range(num_y):
        for k in range(num_x):
            volume[k*step:k*step+outputsize[0],
                                j*step:j*step+outputsize[1],
                                volume.shape[2]-outputsize[2]:] = np.maximum.reduce([volume[k*step:k*step+outputsize[0],
                                j*step:j*step+outputsize[1],
                                volume.shape[2]-outputsize[2]:],ROIs.pop(0)])
        volume[volume.shape[0]-outputsize[0]:,
                            j*step:j*step+outputsize[1],
                            volume.shape[2]-outputsize[2]:] = np.maximum.reduce([volume[volume.shape[0]-outputsize[0]:,
                            j*step:j*step+outputsize[1],
                            volume.shape[2]-outputsize[2]:],ROIs.pop(0)])
    for k in range(num_x):
        volume[k*step:k*step+outputsize[0],
                    volume.shape[1]-outputsize[1]:,
                    volume.shape[2]-outputsize[2]:] = np.maximum.reduce([volume[k*step:k*step+outputsize[0],
                    volume.shape[1]-outputsize[1]:,
                    volume.shape[2]-outputsize[2]:],ROIs.pop(0)])
    volume[volume.shape[0]-outputsize[0]:,
                        volume.shape[1]-outputsize[1]:,
                        volume.shape[2]-outputsize[2]:] = np.maximum.reduce([volume[volume.shape[0]-outputsize[0]:,
                        volume.shape[1]-outputsize[1]:,
                        volume.shape[2]-outputsize[2]:],ROIs.pop(0)])
    return volume

if __name__ == '__name__':
    mask,spacing,origin = nii2array("train/%s/input.nii.gz"%(29))
    rois = cropROI(mask,[512,512,80],32)
    outmask = eraseROI(rois,mask.shape,32).astype(np.uint16)
    outmask = np.transpose(outmask,(2,1,0))
    savenii(outmask,spacing,origin,"test",std=True)