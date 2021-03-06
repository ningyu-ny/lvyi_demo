import sys;
sys.path.append("rgo")
from nii import savenii,nii2array
import numpy as np

def grow(arr,axis,hu,pad,aim=1):
    num=0
    lists = []
    lists.append(axis)
    if aim>=(hu-pad) and aim<=(hu+pad):
        raise Exception("the number of regions up to 1000")
    x,y,z=axis[0],axis[1],axis[2]
    if x>=arr.shape[0] or y>=arr.shape[1] or z>=arr.shape[2] or x<0 or y<0 or z<0:
        return
    while len(lists):
        ax=lists.pop()
        x,y,z=ax[0],ax[1],ax[2]
        if x>=arr.shape[0] or y>=arr.shape[1] or z>=arr.shape[2] or x<0 or y<0 or z<0:
            continue
        if arr[x,y,z]>(hu-pad) and arr[x,y,z]<(hu+pad):
            arr[x,y,z]=aim
            num+=1
            lists.append([x-1,y,z])
            lists.append([x+1,y,z])
            lists.append([x,y-1,z])
            lists.append([x,y+1,z])
            lists.append([x,y,z-1])
            lists.append([x,y,z+1])
    return num

def findbiggest(arr,keep = 1,hu=1000,pad=10):
    outmask=np.zeros(arr.shape)
    for x in range(int(arr.max())):
        mid=np.zeros(arr.shape)
        mid[arr==x+1]=1000
        nums = []
        mask = -1
        # arr[arr!=0]=1000
        for i in range(mid.shape[0]):
            # print("i",i)
            for j in range(mid.shape[1]):
                for k in range(mid.shape[2]):
                    if mid[i,j,k]==hu:#>(hu-pad) and mid[i,j,k]<(hu+pad):
                        nums.append(grow(mid,[i,j,k],hu,pad,aim=mask))
                        mask-=1
        nums_sorted = nums[:]
        nums_sorted.sort(reverse=True)
        for i in range(keep):
            mid[mid==-(nums.index(nums_sorted[i])+1)]=1
        mid[mid<0]=0


        mid[mid!=0]=x+1
        outmask+=mid
        print("%s滤除 %s 个杂质体"%(x+1,len(nums)-keep))
    return outmask

def RegionGrowthOptimize(stlpath,outpath=None,keep = 1):
    if outpath==None:
        outpath=stlpath[:-7]+"_new"
    # print("输入模型名称：",stlpath,",输出模型名称：",outpath+".nii.gz")
    arr,spa,ori = nii2array(stlpath)
    arr = np.array(arr)
    arr = findbiggest(arr,keep)
    savenii(arr,spa,ori,outpath,std=True)

if __name__=="__main__":
    RegionGrowthOptimize("mask.nii.gz")