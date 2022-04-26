import numpy as np
import torch
import argparse
from myloss_wce_group2 import MyLoss
# from openfile import text_create
# from lossFuncs import dice_loss
from torch.utils.data import DataLoader
from torch import autograd, optim
import torch.nn as nn
from torchvision.transforms import transforms
# import vnet_softmax as vnet
from dataset_nii import Vnet_Dataset,make_dataset_train,Vnet_Dataset_test
import matplotlib.pyplot as plt
# from matplotlib import cm
import scipy.misc
import time
from os.path import basename
import os
import pandas as pd
import cv2
# from unet3d_res import unet3d
# from wnet_softmax import wnet as unet3d
import gc
from sklearn.model_selection import KFold
from nii import savenii
from maxregiongrowth import RegionGrowthOptimize as rgo
import random
from rois import eraseROI
import SimpleITK as sitk

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
])

# mask只需要转换为tensor
y_transforms = transforms.Compose([
    transforms.ToTensor(),
])
#参数解析
parse=argparse.ArgumentParser()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()

def mkdir(path):
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        os.makedirs(path)
        print(path+' 创建成功')
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path+' 目录已存在')


def datestr():
    now = time.localtime()
    return '{}{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday)

def timestr():
    now = time.localtime()
    return '{:02}{:02}'.format(now.tm_hour, now.tm_min)

def adjust_learning_rate(optimizer, decay_rate=.95):
    for param_group in optimizer.param_groups:
        if param_group['lr'] > 1e-5:
            param_group['lr'] *= decay_rate

def train_model(train_path,criterion, train_dataloaders,test_dataloaders, k,kfold=5,num_epochs=500,model_path = ''):
    model_path += '%s_%s' %(datestr(),k)
    mkdir(model_path)
    # from unet3d_softmax import unet3d
    # from unet3d_softmax_depth1 import unet3d
    # model = unet3d(1, 4, batch_norm=False, sample=False).to(device)
    # model = vnet.VNet(elu=False, nll=False, num_out=4).to(device)
    # from wnet_softmax import wnet as unet3d
    # model = unet3d(1, 4, batch_norm=True, sample=False).to(device)
    from CEDNet3d import unet3d
    model = unet3d(1, 5, batch_norm=True, sample=False).to(device)
    # model = vnet.VNet(elu=False, nll=False, num_out=4).to(device)
    # model = unet3d(1,4, batch_norm=True, sample=False).to(device)
    # model = unet3d(class_num=4).to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.002)
    model.apply(weights_init)
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    train_loss_list = []
    train_dice_list = []
    epoch_loss_list = []
    test_loss_list = []
    test_dice_list = []
    total_step = 0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        dt_size = len(train_dataloaders.dataset)
        # dt_size_test = len(dataloaders_test.dataset)
        epoch_loss = 0
        epoch_dice=0
        step = 0
        inshape = [88,192, 192]
        for x,x2d,y,y2d in train_dataloaders:
            step += 1
            # mip = torch.detach(x).cpu().numpy().max(2)[0,0]
            # cv2.imshow('mip',(mip/mip.max()*255).astype(np.uint8))
            # cv2.waitKey(0)
            mip = x2d.to(device).float()
            y2d = y2d.to(device)
            #限制前后多余量
            idx = [random.randint(0,x.size()[2]-inshape[0]),random.randint(0,x.size()[3]-inshape[1]),random.randint(20,x.size()[4]-inshape[2]-20)]
            # idx = [10,110,60]#--------------------
            inputs = x[:,:,idx[0]:idx[0]+inshape[0],idx[1]:idx[1]+inshape[1],idx[2]:idx[2]+inshape[2]].to(device).float()
            inputs.requires_grad_()
            labels = y[:,:,idx[0]:idx[0]+inshape[0],idx[1]:idx[1]+inshape[1],idx[2]:idx[2]+inshape[2]].to(device).float()
            optimizer.zero_grad()
            savenii(torch.detach(inputs[0,0]).cpu().numpy(),[1,1,1],[1,1,1],'temp/%sinput'%step,std=True)
            savenii(torch.detach(torch.argmax(labels[0], 0)).cpu().numpy(),[1,1,1],[1,1,1],'temp/%slabel'%step,std=True)

            out2d,out3d = model(mip,inputs,idx)
            savenii(torch.detach(torch.argmax(out3d[0], 0)).cpu().numpy(),[1,1,1],[1,1,1],'temp/%spredict'%step,std=True)
            loss,dices = criterion(out2d,out3d,labels,y2d)
            # if epoch==20:
            #     temp = torch.detach(out2d).cpu().numpy()[0]
            #     temp = np.argmax(temp, 0)
            #     cv2.imshow('temp',(temp/temp.max()*255).astype(np.uint8))
            #     cv2.waitKey(0)
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.item())
            # train_dice_list.append(dices)
            epoch_loss += loss.item()
            epoch_dice += dices
            print("%d/%d,train_loss:%0.6f" % (step, (dt_size - 1) // train_dataloaders.batch_size + 1, loss.item()))
        
        test_loss=0
        test_dice=0
        # model.eval()
        step_test=0
        with torch.no_grad():
            for x,x2d,y,y2d in test_dataloaders:#
                step_test += 1
                mip = x2d.to(device).float()
                y2d = y2d.to(device)
                idx = [random.randint(0,x.size()[2]-inshape[0]),random.randint(0,x.size()[3]-inshape[1]),random.randint(0,x.size()[4]-inshape[2])]
                # idx = [10,110,60]#----------------
                inputs = x[:,:,idx[0]:idx[0]+inshape[0],idx[1]:idx[1]+inshape[1],idx[2]:idx[2]+inshape[2]].to(device).float()
                labels = y[:,:,idx[0]:idx[0]+inshape[0],idx[1]:idx[1]+inshape[1],idx[2]:idx[2]+inshape[2]].to(device).float()
                optimizer.zero_grad()

                out2d,out3d = model(mip,inputs,idx)
                loss,dices = criterion(out2d,out3d,labels,y2d)
                # loss.backward()
                test_loss+=loss.item()
                test_dice += dices
                print("%d/%d,test_loss:%0.6f" % (step_test, (len(test_dataloaders.dataset) - 1) // test_dataloaders.batch_size + 1, loss.item()))
        epoch_dice/=(len(train_dataloaders.dataset)/train_dataloaders.batch_size)
        test_dice/=(len(test_dataloaders.dataset)/test_dataloaders.batch_size)
        model.train()
        epoch_loss_list.append(epoch_loss)
        train_dice_list.append(epoch_dice.tolist())
        test_loss_list.append(test_loss*(kfold-1))
        test_dice_list.append(test_dice.tolist())
        step_loss = pd.DataFrame({'step': range(len(train_loss_list)), 'step_loss': train_loss_list})
        step_loss.to_csv(model_path + '/' + 'step_loss.csv',index=False)
        # adjust_learning_rate(optimizer)
        print("epoch %d loss:%0.3f,test_loss:%0.3f" % ((epoch+1), epoch_loss, test_loss*(kfold-1)))
        if epoch % 5 == 4:
            torch.save(model.state_dict(), (model_path + '/%s_epoch_%d.pth' %(timestr(),(epoch+1))))
    plt.plot(epoch_loss_list,label="train")
    plt.plot(test_loss_list,label="test")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(bbox_to_anchor=(0.8, 0.97), loc=2, borderaxespad=0.)
    plt.savefig(model_path+"/accuracy_loss%s.jpg"%k)
    plt.close()
    train_dice_list = np.array(train_dice_list)
    test_dice_list = np.array(test_dice_list)
    ifnotsave=1
    while ifnotsave:
        try:
            step_dice = pd.DataFrame({'step': range(len(epoch_loss_list)), 
            'train_loss': epoch_loss_list, 'test_loss': test_loss_list,
            'train_dice_1': train_dice_list[:,0], 'train_dice_2': train_dice_list[:,1],
            'train_dice_3': train_dice_list[:,2],
            'test_dice_1': test_dice_list[:,0],
            'test_dice_2': test_dice_list[:,1],'test_dice_3': test_dice_list[:,2]
            })
            step_dice.to_csv(model_path + '/' + 'epoch_dice.csv',index=False)
            ifnotsave=0
        except:
            input("保存失败，按任意键重试")

    # text_create(model_path+"/fold_%s_train"%k,epoch_loss_list)
    # text_create(model_path+"/fold_%s_test"%k,test_loss_list)
    # text_create(model_path+"/fold_%s_traindice"%k,train_dice_list)
    # text_create(model_path+"/fold_%s_testdice"%k,test_dice_list)
    del optimizer
    del model
    return

#训练模型
def train(train_path="trainset/train9_kidney0tumor1_64_tumoronly_cut/data",train_path_std="",model_path = "",test_path=""):

    batch_size = 1
    criterion = MyLoss()
    datasets=make_dataset_train(train_path)
    datasets_std = make_dataset_train(train_path_std)
    kfold = 5
    kf = KFold(n_splits=kfold)
    k=0
    for trainsets, testsets in kf.split(datasets):
        # if k==0:
        #     k+=1
        #     continue
        trainsets = np.array(datasets)[trainsets].tolist()
        testsets = np.array(datasets)[testsets].tolist()
        testsets_std = datasets_std[int(k * len(datasets_std) / kfold):int((k + 1) * len(datasets_std) / kfold)]
        train_data = Vnet_Dataset(trainsets,transform=x_transforms,mask_transform=y_transforms)
        train_dataloaders = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=0)
        test_data = Vnet_Dataset(testsets,transform=x_transforms,mask_transform=y_transforms)
        test_dataloaders = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

        train_model(train_path,criterion, train_dataloaders,test_dataloaders, k, kfold,model_path=model_path)
        k+=1

def segonly(test_path="datapro/seg",model_path='model/goodmodels/400k1t1_bce_bs1_dice0963.pth', mod="wnet"):
    from CEDNet3d import unet3d
    model = unet3d(1, 5, batch_norm=True, sample=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage.cuda(0)))
    datasets=make_dataset_train(test_path)
    test_data = Vnet_Dataset_test(datasets,transform=x_transforms,mask_transform=y_transforms)
    test_dataloaders = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    # model.eval()
    i=0
    for oriimage,images,idxs,spc,ori,x_path,input_shape0,input_shape1 in test_dataloaders:
        out_images = []
        spc=(spc[0].item(),spc[1].item(),spc[2].item())
        ori=(ori[0].item(),ori[1].item(),ori[2].item())
        for i,image0 in enumerate(images):
            image = torch.tensor(image0, dtype=torch.float32).unsqueeze(0).to(device)
            mip = oriimage.max(1).values.to(device).float().unsqueeze(0)
            idx = idxs[i]
            y2d,y3d = model(mip,image,idx)
            # folder_path = save_path + '/' + folderlist[i]
            y = y3d.view((5,88,192, 192))
            # image_np=(image_np.cpu().numpy()*255).astype(np.uint8)
            y = torch.argmax(y, 0)
            out_images.append(torch.detach(y).cpu().numpy())
            # savenii(torch.detach(image0).cpu().numpy()[0],spc,ori,x_path[0][:-7]+'predict%s'%i+"cut",std=True)
            # savenii(out_images[-1],spc,ori,x_path[0][:-7]+'predict%s'%i,std=True)
            
            print('Finish:%s/%s'%(i,len(images)))
        out = eraseROI(out_images,input_shape1,60)
        out1 = np.zeros(input_shape0)
        out1[:out.shape[0],:out.shape[1],:out.shape[2]] = out

        savenii(out1,spc,ori,x_path[0][:-7]+'predict',std=True)
        rgo('%spredict.nii.gz'%x_path[0][:-7])
        i+=1

if __name__ == '__main__':
    #训练用
    train_path = "data/train"
    model_path ='model/cednet/w1d01'
    train(train_path,model_path)
    #测试用
    niipath = "data/eval"
    modelpath = "model/1919_epoch_125.pth"
    segonly(niipath,modelpath)