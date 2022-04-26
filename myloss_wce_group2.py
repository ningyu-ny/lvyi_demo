import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, out2d,out3d,labels,labels_2d):
        # N = target.size(0)
        # print(N)
        smooth = 1
        q = out3d
        p = labels
        weight=torch.tensor([[[[0.1]]],[[[1]]],[[[3]]],[[[1]]],[[[3]]]]).to(device)
        weight_d=torch.tensor([[[[0.1]]],[[[1]]],[[[1]]],[[[1]]],[[[1]]]]).to(device)
        min = torch.tensor([0.00001]).to(device)
        wce = -p * torch.log(q.max(min)) * weight#WCE
        # x2 = p*(1 - 2*q + q*q)+0.1*(1-p)*q*q#X2
        # myloss = middle.sum()/input_flat.sum()
        
        # myloss = middle.sum()
        intersection = q * p * weight_d
        dice_weight = (2 * intersection[:,1:5].sum() + smooth) / ((q* weight_d)[:,1:5].sum() + (p* weight_d)[:,1:5].sum() + smooth)
        # dice_mean=(2 * (p*q)[:,1:5].sum() + smooth) / (q[:,1:5].sum() + p[:,1:5].sum() + smooth)
        dice_mean=(2 * (p*q).sum() + smooth) / (q.sum() + p.sum() + smooth)
        diceloss = 1 - dice_mean

        q_2d = out2d
        p_2d = labels_2d
        # dice_mean_2d=(2 * (p_2d*q_2d)[:,1:5].sum() + smooth) / (q_2d[:,1:5].sum() + p_2d[:,1:5].sum() + smooth)#论文没说是全标签dice还是目标dice
        dice_mean_2d=(2 * (p_2d*q_2d).sum() + smooth) / (q_2d.sum() + p_2d.sum() + smooth)#论文没说是全标签dice还是目标dice
        diceloss_2d = 1 - dice_mean_2d

        threshold=0.5
        
        q_dice=q+0.0
        q_dice[q_dice>threshold]=1
        q_dice[q_dice<=threshold]=0
        dice1=(2 * (p*q_dice)[:,1].sum() + smooth) / (q_dice[:,1].sum() + p[:,1].sum() + smooth)
        dice2=(2 * (p*q_dice)[:,2].sum() + smooth) / (q_dice[:,2].sum() + p[:,2].sum() + smooth)
        dice3=(2 * (p*q_dice)[:,3].sum() + smooth) / (q_dice[:,3].sum() + p[:,3].sum() + smooth)
        dice4=(2 * (p*q_dice)[:,4].sum() + smooth) / (q_dice[:,4].sum() + p[:,4].sum() + smooth)
        # dice4=(2 * (p*q_dice)[:,4].sum() + smooth) / (q_dice[:,4].sum() + p[:,4].sum() + smooth)
        myloss = 0.5*diceloss + 0.5*diceloss_2d + torch.mean(wce)
        print("WBCE:%04f"%torch.mean(wce).item(),"Dice3D:%04f "%dice_mean.item(),"Dice2D:%04f "%dice_mean_2d.item())
        print("3D: Dice1:%04f"%dice1.item(),"Dice2:%04f"%dice2.item(),"Dice3:%04f"%dice3.item(),"Dice4:%04f"%dice4.item())
        dices=np.array([dice1.item(),dice2.item(),dice3.item(),dice4.item()])
        return myloss,dices

class MyLossx(nn.Module):
    def __init__(self):
        super(MyLossx, self).__init__()

    def    forward(self, input, target):
        N = target.size(0)
        # print(N)
        smooth = 1
 
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        middle = input_flat * 2 * (0.5 - target_flat) + target_flat
        myloss = torch.mean(middle)
        # intersection = input_flat * target_flat
        # print(intersection.sum(0))
        # print(intersection.sum(1))
        # dice = (2 * intersection.sum() + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
        # diceloss = 1 - dice

        return myloss


class Dice(nn.Module):
    def __init__(self):
            super(Dice, self).__init__()

    def    forward(self, input, target):
        N = target.size(0)
        # print(N)
        smooth = 1
 
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)


        intersection = input_flat * target_flat
        # print(intersection.sum(0))
        # print(intersection.sum(1))
        dice = (2 * intersection.sum()) / (input_flat.sum() + target_flat.sum())
        # diceloss = 1 - dice

        return dice
