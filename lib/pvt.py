import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np
from lib.pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.Decoder import HGMM, BMD
                  

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out





class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    
class BasicConv2dV2(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=32):
        super(BasicConv2dV2, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False, groups=groups)
        self.conv2 = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.conv2(x)
        x = self.bn(x1+x2)
        return x


class CFM(nn.Module):
    def __init__(self, channel):
        super(CFM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1 #x4
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1

class CFMV2(nn.Module):
    def __init__(self, channel):
        super(CFMV2, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=2,dilation=2)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=4,dilation=4)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=2,dilation=2)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=2,dilation=2)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=2,dilation=2)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1 #x4
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1

class CFMV3(nn.Module):
    def __init__(self, channel):
        super(CFMV3, self).__init__()
        self.relu = nn.ReLU(True)

        self.conv1_1 = BasicConv2d(channel, channel //2, (3,3),padding = 1,dilation=(1,1))
        self.conv1_2 = BasicConv2d(channel //2, channel //4, (3,3),padding = 2,dilation=(2,2))
        self.conv1_3 = BasicConv2d(channel //4, channel //4, (3,3),padding = 3,dilation=(3,3))

        self.conv2_1 = BasicConv2d(channel, channel //2, (3,3),padding = 1,dilation=(1,1))
        self.conv2_2 = BasicConv2d(channel //2, channel //4, (3,3),padding = 2,dilation=(2,2))
        self.conv2_3 = BasicConv2d(channel //4, channel //4, (3,3),padding = 3,dilation=(3,3))

        self.conv3_1 = BasicConv2d(channel, channel //2, (3,3),padding = 1,dilation=(1,1))
        self.conv3_2 = BasicConv2d(channel //2, channel //4, (3,3),padding = 3,dilation=(3,3))
        self.conv3_3 = BasicConv2d(channel //4, channel //4, (3,3),padding = 5,dilation=(5,5))

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3):

        o1_1 = self.conv1_1(x1)
        o1_2 = self.conv1_2(o1_1)
        o1_3 = self.conv1_3(o1_2)
        x1 =  torch.cat([o1_1,o1_2,o1_3],1)

        o2_1 = self.conv2_1(x2)
        o2_2 = self.conv2_2(o2_1)
        o2_3 = self.conv2_3(o2_2)
        x2 =  torch.cat([o2_1,o2_2,o2_3],1)

        o3_1 = self.conv3_1(x3)
        o3_2 = self.conv3_2(o3_1)
        o3_3 = self.conv3_3(o3_2)
        x3 =  torch.cat([o3_1,o3_2,o3_3],1)

        x1_1 = x1 #x4
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1
    
class CFMV4(nn.Module):
    def __init__(self, channel):
        super(CFMV4, self).__init__()
        self.relu = nn.ReLU(True)

        self.conv1_1 = BasicConv2d(channel, channel //2, (3,3),padding = 1,dilation=(1,1))
        self.conv1_2 = BasicConv2d(channel //2, channel //4, (3,3),padding = 2,dilation=(2,2))
        self.conv1_3 = BasicConv2d(channel //4, channel //4, (3,3),padding = 3,dilation=(3,3))

        self.conv2_1 = BasicConv2d(channel, channel //2, (3,3),padding = 1,dilation=(1,1))
        self.conv2_2 = BasicConv2d(channel //2, channel //4, (3,3),padding = 2,dilation=(2,2))
        self.conv2_3 = BasicConv2d(channel //4, channel //4, (3,3),padding = 3,dilation=(3,3))

        self.conv3_1 = BasicConv2d(channel, channel //2, (3,3),padding = 1,dilation=(1,1))
        self.conv3_2 = BasicConv2d(channel //2, channel //4, (3,3),padding = 3,dilation=(3,3))
        self.conv3_3 = BasicConv2d(channel //4, channel //4, (3,3),padding = 5,dilation=(5,5))

        self.conv4_1 = BasicConv2d(channel, channel //2, (3,3),padding = 1,dilation=(1,1))
        self.conv4_2 = BasicConv2d(channel //2, channel //4, (3,3),padding = 3,dilation=(3,3))
        self.conv4_3 = BasicConv2d(channel //4, channel //4, (3,3),padding = 5,dilation=(5,5))

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_upsample41 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample42 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample43 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4 * channel, 4 * channel, 3, padding=1)



        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)

        self.conv4 = BasicConv2d(4 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3, x4):

        o1_1 = self.conv1_1(x1)
        o1_2 = self.conv1_2(o1_1)
        o1_3 = self.conv1_3(o1_2)
        x1 = torch.cat([o1_1,o1_2,o1_3],1)

        o2_1 = self.conv2_1(x2)
        o2_2 = self.conv2_2(o2_1)
        o2_3 = self.conv2_3(o2_2)
        x2 = torch.cat([o2_1,o2_2,o2_3],1)

        o3_1 = self.conv3_1(x3)
        o3_2 = self.conv3_2(o3_1)
        o3_3 = self.conv3_3(o3_2)
        x3 = torch.cat([o3_1,o3_2,o3_3],1)

        o4_1 = self.conv4_1(x4)
        o4_2 = self.conv4_2(o4_1)
        o4_3 = self.conv4_3(o4_2)
        x4 = torch.cat([o4_1,o4_2,o4_3],1)

        x1_1 = x1 #x4
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3
        x4_1 = self.conv_upsample41(self.upsample(self.upsample(self.upsample(x1)))) \
               * self.conv_upsample42(self.upsample(self.upsample(x2))) \
                * self.conv_upsample43(self.upsample(x3)) * x4
        #print(x4_1.shape)

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)
       # print(x3_2.shape)

        x4_2 = torch.cat((x4_1, self.conv_upsample6(self.upsample(x3_2))), 1)
        x4_2 = self.conv_concat4(x4_2)

        x1 = self.conv4(x4_2)

        return x1

class CFMV5(nn.Module):
    def __init__(self, channel):
        super(CFMV5, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()


    def forward(self, x1, x2, x3):
        x1_1 = x1 #x4
        # x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        # x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
        #        * self.conv_upsample3(self.upsample(x2)) * x3
        x2_1 = self.relu1(self.conv_upsample1(self.upsample(x1))) * x2
        x3_1 = self.relu2(self.conv_upsample2(self.upsample(self.upsample(x1)))) \
               * self.relu3(self.conv_upsample3(self.upsample(x2))) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1

class CFMV6(nn.Module):
    def __init__(self, channel):
        super(CFMV6, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3, x4):
        x1_1 = x1 #x4
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3 * x4

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1

class CFMV7(nn.Module):
    def __init__(self, channel):
        super(CFMV7, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2dV2(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2dV2(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2dV2(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2dV2(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2dV2(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2dV2(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2dV2(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2dV2(3 * channel, channel, 3, padding=1, groups=channel)

    def forward(self, x1, x2, x3):
        x1_1 = x1 #x4
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1

class CFMV8(nn.Module):
    def __init__(self, channel):
        super(CFMV8, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

        self.deconv1 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(2 * channel, 2 * channel, kernel_size=4, stride=2, padding=1)


    def forward(self, x1, x2, x3):
        x1_1 = x1 #x4
        x2_1 = self.conv_upsample1(self.deconv1(x1)) * x2
        x3_1 = self.conv_upsample2(self.deconv1(self.deconv1(x1))) \
               * self.conv_upsample3(self.deconv2(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.deconv3(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.deconv4(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1

class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


def erosion_to_dilate(output):
        output_device = output.device
        z = output.cpu().detach().numpy()
        z = np.where(z > 0.3, 1.0, 0.0)  #covert segmentation result
        z = torch.tensor(z)    
        kernel = np.ones((4, 4), np.uint8)   # kernal matrix
        maskd = np.zeros_like(output.cpu().detach().numpy())  #result array
        maske = np.zeros_like(output.cpu().detach().numpy())  #result array
        background = np.zeros_like(output.cpu().detach().numpy())  #result array
        for i in range(output.shape[0]):
            y = z[i].permute(1,2,0)
            erosion = y.cpu().detach().numpy()
            dilate = y.cpu().detach().numpy()
            dilate = np.array(dilate,dtype='uint8')
            erosion = np.array(erosion,dtype='uint8')
            erosion = cv.erode(erosion, kernel, 4)  
            dilate = cv.dilate(dilate, kernel, 4)
            mask1 = torch.tensor(dilate-erosion).unsqueeze(-1).permute(2,0,1)
            mask2 = torch.tensor(erosion).unsqueeze(-1).permute(2,0,1)
            maskd[i] = mask1
            maske[i] = mask2
            background[i] = torch.tensor(1-dilate).unsqueeze(-1).permute(2,0,1)
        maskd = torch.tensor(maskd,device=output_device)#边界
        maske = torch.tensor(maske,device=output_device)#内部  
        background = torch.tensor(background,device=output_device)#背景    
        return torch.cat([maske, background, maskd], dim=1)


class BDFM(nn.Module):
    def __init__(self, in_channels):
        super(BDFM, self).__init__()
        self.relu = nn.ReLU()
        self.conv1x1_feature = BasicConv2d(in_channels, in_channels, 1)
        self.conv1x1_mid_feature_1 = BasicConv2d(in_channels, in_channels, 1)
        self.conv1x1_mid_feature_2 = BasicConv2d(in_channels, in_channels, 1)
        self.conv1x1_output = BasicConv2d(in_channels*2, in_channels, 1)

    def forward(self, feature, m):

        B,C,H,W = feature.shape

        m = self.relu(m)  # [B, 1, H, W]

        fbu = erosion_to_dilate(m)  # [B, 3, H, W]

        fbu = fbu.reshape(B, 3 , H*W)  # [B, 3, H*W]

        feature_1 = feature.reshape(B, C, H*W)  # [B, C, H*W]

        feature_1 = feature_1.permute(0, 2, 1)  # [B, H*W, C]

        mid_feature = torch.matmul(fbu, feature_1)  # [B, 3, C]

        conv_feature = self.conv1x1_feature(feature)  # [B, C, H, W]
        conv_feature = conv_feature.reshape(B, C, H*W)  # [B, C, H*W]

        mid_feature_1 = torch.matmul(mid_feature, conv_feature)  # [B, 3, H*W]
        mid_feature_2 = torch.matmul(mid_feature.permute(0, 2, 1), mid_feature_1)  # [B, C, H*W]

        mid_feature_2 = mid_feature_2.reshape(B, C, H, W)  # [B, C, H, W]

        output = torch.cat((feature,mid_feature_2),dim=1)  # [B, C, H, W]

        output = self.conv1x1_output(output)  # [B, C, H, W]

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
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

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

class multiscale_feature_aggregation(nn.Module):
    def __init__(self, in_c, out_c):#[320,128,64],64
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.up_1 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up_2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_2_1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.c1 = conv2d(in_c[0], out_c, kernel_size=1, padding=0)
        self.c2 = conv2d(in_c[1], out_c, kernel_size=1, padding=0)
        self.c3 = conv2d(in_c[2], out_c, kernel_size=1, padding=0)
        

        self.c12_11=conv2d(out_c*2, out_c)
        self.c12_12=conv2d(out_c, out_c, kernel_size=1, padding=0)

        self.c12_21=conv2d(out_c*2, out_c)
        self.c12_22=conv2d(out_c, out_c, kernel_size=1, padding=0)

        self.c22 = conv2d(2*out_c, out_c)
        self.c23 = conv2d(out_c, out_c)

    def forward(self, x1, x2, x3):
        x1 = self.up_1(x1)# 扩大4倍
        x1 = self.c1(x1) 

        x2 = self.c2(x2)
        x2=self.up_2(x2) # 扩大2倍
        x12=torch.cat([x1,x2],1)
        x12=self.up_2_1(x12) # 再扩大2倍
        
        x12_1=self.c12_11(x12)
        x12_1=self.c12_12(x12_1)
        
        x12_2=self.c12_21(x12)
        x12_2=self.c12_22(x12_2)

        x3=self.up_3(x3)
        x3_1 = self.c3(x3)
        x3_1=(x3_1*x12_1)+x12_2
        x=torch.cat([x3_1,x3],1)
        x=self.c23(self.c22(x))
        return x

class HGM(nn.Module):
    def __init__(self, channel=32,depths=[1,1],device=None):
        super(HGM, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        self.CFM = CFM(channel)

        self.hgmm = HGMM(dims=[64,64],depths=depths,device=device)
        self.decoder = BMD()
        self.hgmm02 = HGMM(dims=[32,32],depths=depths,device=device)
        self.hgmm03 = HGMM(dims=[32,32],depths=depths,device=device)

        self.bdfm = BDFM(in_channels=32)
        
        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out_CFM = nn.Conv2d(channel, 1, 1)

    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        
        # CIM
        lm_feature = self.hgmm(x1)

        # CFM
        x2_t = self.Translayer2_1(x2)  
        x3_t = self.Translayer3_1(x3)  
        x4_t = self.Translayer4_1(x4)  
        cfm_feature = self.CFM(x4_t, x3_t, x2_t)

        # UALM_V2
        lm_middle_output = self.Translayer2_0(lm_feature)
        lm_middle_output = self.down05(lm_middle_output)
        mid_feature = self.out_CFM(cfm_feature)
        UA_feature = self.bdfm(lm_middle_output,mid_feature) + lm_middle_output
        UA_feature = self.hgmm02(UA_feature)

        cfm_feature = self.hgmm03(cfm_feature)

        x12 = []
        x12.append(x1)
        x12.append(x2)
        prediction = cfm_feature + UA_feature
        prediction = self.decoder(x12,prediction)
        
        return prediction


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HGM()
    opttrainsize = 352
    input_tensor = torch.randn(1, 3, opttrainsize, opttrainsize).to(device)
    model.to(device=device)

    size_rates = [0.75, 1, 1.25]
    for rate in size_rates:
        trainsize = int(round(opttrainsize*rate/32)*32)
        if rate != 1:
            images = F.upsample(input_tensor, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            res1 = model(images)
            print("input: {}".format(images.shape))
            print("output: {}".format(res1.shape))
        else:
            images = input_tensor
            res1 = model(images)
            print("input: {}".format(images.shape))
            print("output: {}".format(res1.shape))

