import torch.utils.data as data
import PIL.Image as Image
import os
from glob import glob
import cv2
import numpy as np
import torch
import SimpleITK as sitk
from nii import nii2array
from rois import cropROI

def make_dataset_train(root):
    datasets = []
    for dirName,subdirList,fileList in os.walk(root):
        data_filelist = []
        # mask_filelist = []
        for filename in fileList:
            if "image" in filename.lower() and "nii" in filename.lower() and "predict" not in filename.lower(): #判断文件是否为dicom文件
                filepth = os.path.join(dirName,filename)
                maskpth = filepth.replace("image","labels")
                data_filelist.append([[filepth],[maskpth]]) # 加入到列表中
            # if "mask.nii.gz" in filename.lower(): #判断文件是否为dicom文件
                # data_filelist.append([os.path.join(dirName,filename)]) # 加入到列表中

        if len(data_filelist)<1:
            continue
        datasets.extend(data_filelist)
    return datasets

def merge_images_train(files):
    image_depth = len(files)
    image_sample = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    image_height, image_width = image_sample.shape
    image_3d = np.empty((image_depth,image_height, image_width))
    index = 0
    for file in files:
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        image_3d[index, :, :] = image
        index += 1
    return image_3d

class Vnet_Dataset(data.Dataset):
    def __init__(self, datasets, transform=None, mask_transform=None):
        self.datasets = datasets
        self.transform = transform
        self.mask_transform = mask_transform
        # self.classnum = classnum

    def __getitem__(self, index):
        x_path = self.datasets[index][0]
        y_path = self.datasets[index][1]
        image = sitk.ReadImage(x_path[0])
        image = torch.tensor((sitk.GetArrayFromImage(image).astype(np.int32)), dtype=torch.float32).unsqueeze(0)
        sizenew = [image.size()[1]//8*8,image.size()[2]//8*8,image.size()[3]//8*8]
        mask = sitk.ReadImage(y_path[0])
        mask = torch.tensor(sitk.GetArrayFromImage(mask), dtype=torch.long)
        image = image[:,0:sizenew[0],0:sizenew[1],0:sizenew[2]]
        mask = mask[0:sizenew[0],0:sizenew[1],0:sizenew[2]]
        # classnum = self.classnum
        classnum = mask.max().item()
        image2d = image.max(1).values
        mask2d = torch.zeros((int(classnum)+1, mask.shape[1], mask.shape[2]), dtype=torch.short).scatter_(0, mask.max(0).values.unsqueeze(0), 1)
        mask3d = torch.zeros((int(classnum)+1, mask.shape[0], mask.shape[1], mask.shape[2]), dtype=torch.short).scatter_(0, mask.unsqueeze(0), 1)
        
        # print(mask.shape)
            # for i in range(64):
            #     sample = mask[:,:,i].numpy()
            #     cv2.imwrite('sample/%d.bmp'%i,sample)
            #     print('sample/%d.bmp'%i)
            # print(2)
        return image,image2d, mask3d,mask2d

    def __len__(self):
        return len(self.datasets)

class Vnet_Dataset_test(data.Dataset):
    def __init__(self, datasets, transform=None, mask_transform=None):
        self.datasets = datasets
        self.transform = transform
        self.mask_transform = mask_transform

    def __getitem__(self, index):
        x_path = self.datasets[index][0]
        array_size = np.array((88,192, 192))
        image,spc,ori = nii2array(x_path[0])
        sizenew = [image.shape[0]//8*8,image.shape[1]//8*8,image.shape[2]//8*8]
        image1 = image[0:sizenew[0],0:sizenew[1],0:sizenew[2]]
        rois_arrays,idxs = cropROI(image1.astype(np.int32),array_size,60)
        # image = torch.tensor(image.astype(np.int32), dtype=torch.float32).unsqueeze(0)#暂时删除
        # if self.transform is not None:
        # image = (image/image.max()*255).astype(np.uint8)
        # image = self.transform(image)
        # height, width, depth = image.shape
        # image = image.reshape(1, width, depth,height)
    # if self.mask_transform is not None:

        return image1,rois_arrays,idxs,spc,ori,x_path[0],image.shape,image1.shape

    def __len__(self):
        return len(self.datasets)
