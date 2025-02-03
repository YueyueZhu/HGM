import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from lib.PSA import PSA_p

class conv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        
        self.bn_acti = bn_acti
        
        self.conv = nn.Conv2d(nIn, nOut, kernel_size = kSize,
                              stride=stride, padding=padding,
                              dilation=dilation,groups=groups,bias=bias)
        
        if self.bn_acti:
            self.bn_relu = BNPReLU(nOut)
            
    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_relu(output)

        return output  
     
class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)
        
        return output

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class FSEMV2(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.bn_relu_1 = BNPReLU(nIn)
        self.bn_relu_2 = BNPReLU(nIn)
        self.bn1=nn.BatchNorm2d(nIn)

        self.psa1=PSA_p(nIn,nIn)

        self.conv2_1 =conv2d(nIn, nIn//2, (3,3),padding = 1,dilation=(1,1),act=False)
        self.conv2_2 =conv2d(nIn//2, nIn//2, (3,3),padding = 1,dilation=(1,1),act=False)

        self.conv3_1 =conv2d(nIn, nIn //2, (3,3),padding = 1,dilation=(1,1),act=False)
        self.conv3_2 =conv2d(nIn //2, nIn //4, (3,3),padding = 2,dilation=(2,2),act=False)
        self.conv3_3 =conv2d(nIn //4, nIn //4, (3,3),padding = 3,dilation=(3,3),act=False)

        self.conv4_1 =conv2d(nIn, nIn //2, (3,3),padding = 1,dilation=(1,1),act=False)
        self.conv4_2 =conv2d(nIn //2, nIn //4, (3,3),padding = 3,dilation=(3,3),act=False)
        self.conv4_3 =conv2d(nIn //4, nIn //4, (3,3),padding = 5,dilation=(5,5),act=False)

        self.conv_out=conv2d(nIn,nOut)
        self.selayer = SELayer(nIn)
        

    def forward(self, x):
        o1_2 = self.psa1(x)
        
        o2_1 = self.conv2_1(x)
        o2_2 = self.conv2_2(o2_1)
        
        o3_1 = self.conv3_1(x)
        o3_2 = self.conv3_2(o3_1)
        o3_3 = self.conv3_3(o3_2)

        o4_1 = self.conv4_1(x)
        o4_2 = self.conv4_2(o4_1)
        o4_3 = self.conv4_3(o4_2)
        
        o4=torch.cat([o4_1,o4_2,o4_3],1)
        o3_4=torch.cat([o3_1,o3_2,o3_3],1)
        o2_3=torch.cat([o2_1,o2_2],1)

        x_out=self.bn1(o4+o3_4+o2_3)
        x_out=self.selayer(x_out)+o1_2
        x_out=self.conv_out(x_out)

        return x_out
