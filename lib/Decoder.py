import os
from re import S
from xml.dom import xmlbuilder
import numpy as np
import torch
from torch import nn
from typing import Union, Type, List, Tuple

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder

from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks

from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from torch.cuda.amp import autocast
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from mamba_ssm import Mamba,Mamba2
import copy
from lib.FSEM import FSEMV2
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
    
    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

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
    
class BiPixelMambaLayer(nn.Module):
    def __init__(self, dim, p, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)


        self.mamba_forw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                use_fast_path=False,
        )

     
        self.out_proj = copy.deepcopy(self.mamba_forw.out_proj)


        
        self.mamba_backw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                use_fast_path=False,

        )

      
        # adjust the window size here to fit the feature map
        self.p = p*5
        self.p1 = 5*p
        self.p2 = 7*p
        self.p3 = 6*p
        self.mamba_forw.out_proj = nn.Identity()
        self.mamba_backw.out_proj = nn.Identity()
       


    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        ll = len(x.shape)

        B, C = x.shape[:2]

        assert C == self.dim
        img_dims = x.shape[2:]

        if ll == 5: #3d
         
            Z,H,W = x.shape[2:]

            if Z%self.p1==0 and H%self.p2==0 and W%self.p3==0:
                x_div = x.reshape(B, C, Z//self.p1, self.p1, H//self.p2, self.p2, W//self.p3, self.p3)
                x_div = x_div.permute(0, 3, 5, 7, 1, 2, 4, 6).contiguous().view(B*self.p1*self.p2*self.p3, C, Z//self.p1, H//self.p2, W//self.p3)
            else:
                x_div = x

        elif ll == 4: #2d
            H,W = x.shape[2:]

            if H%self.p==0 and W%self.p==0:                
                x_div = x.reshape(B, C, H//self.p, self.p, W//self.p, self.p).permute(0, 3, 5, 1, 2, 4).contiguous().view(B*self.p*self.p, C, H//self.p, W//self.p)            
            else:
                x_div = x
        

        NB = x_div.shape[0]
        if ll == 5: #3d
            NZ,NH,NW = x_div.shape[2:]
        else:
            NH,NW = x_div.shape[2:]

        n_tokens = x_div.shape[2:].numel()
   

        x_flat = x_div.reshape(NB, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)#在通道上进行归一化

        y_norm = torch.flip(x_norm, dims=[1])

        x_mamba = self.mamba_forw(x_norm)
        y_mamba = self.mamba_backw(y_norm)

        x_out = self.out_proj(x_mamba + torch.flip(y_mamba, dims=[1]))

        if ll == 5:
            if Z%self.p1==0 and H%self.p2==0 and W%self.p3==0:
                x_out = x_out.transpose(-1, -2).reshape(B, self.p1, self.p2, self.p3, C, NZ, NH, NW).permute(0, 4, 5, 1, 6, 2, 7, 3).contiguous().reshape(B, C, *img_dims)
            else:
                x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)
        if ll == 4:
            if H%self.p==0 and W%self.p==0:
                x_out = x_out.transpose(-1, -2).reshape(B, self.p, self.p, C, NH, NW).permute(0, 3, 4, 1, 5, 2).contiguous().reshape(B, C, *img_dims)
            else:
                x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)
        out = x_out + x

        return out

class BiWindowMambaLayer(nn.Module):
    def __init__(self, dim, p, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        self.p = p
        self.mamba_forw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                use_fast_path=False,
        )

     
        self.out_proj = copy.deepcopy(self.mamba_forw.out_proj)

        

        
        self.mamba_backw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                use_fast_path=False,

        )

      

       
        self.mamba_forw.out_proj = nn.Identity()
        self.mamba_backw.out_proj = nn.Identity()
       


    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        ll = len(x.shape)
     
       

        B, C = x.shape[:2]

        assert C == self.dim
   
        img_dims = x.shape[2:]



        if ll == 5: #3d
            
            Z,H,W = x.shape[2:]

            if Z%self.p==0 and H%self.p==0 and W%self.p==0:
                pool_layer = nn.AvgPool3d(self.p, stride=self.p)
                x_div = pool_layer(x)
            else:
                x_div = x

        elif ll == 4: #2d

            H,W = x.shape[2:]
            if self.p==0:
                self.p = 1
            if H%self.p==0 and W%self.p==0:
                pool_layer = nn.AvgPool2d(self.p, stride=self.p)
                x_div = pool_layer(x)
            else:
                x_div = x
        

      
        if ll == 5: #3d
            NZ,NH,NW = x_div.shape[2:]
        else:
            NH,NW = x_div.shape[2:]

        n_tokens = x_div.shape[2:].numel()
   

        x_flat = x_div.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        y_norm = torch.flip(x_norm, dims=[1])

        x_mamba = self.mamba_forw(x_norm)
        y_mamba = self.mamba_backw(y_norm)

        x_out = self.out_proj(x_mamba + torch.flip(y_mamba, dims=[1]))

        if ll == 5:
            if Z%self.p==0 and H%self.p==0 and W%self.p==0:
                unpool_layer = nn.Upsample(scale_factor=self.p, mode='nearest')
                x_out = x_out.transpose(-1, -2).reshape(B, C, NZ, NH, NW)
                x_out = unpool_layer(x_out)
            else:
                x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)
        if ll == 4:
            if self.p==0:
                self.p = 1
            if H%self.p==0 and W%self.p==0:
                unpool_layer = nn.Upsample(scale_factor=self.p, mode='nearest')
                x_out = x_out.transpose(-1, -2).reshape(B, C, NH, NW)
                x_out = unpool_layer(x_out)
            else:
                x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)
                
        out = x_out + x

        return out

class BiWindowMambaLayerV2(nn.Module):
    def __init__(self, dim, p, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        self.p = p
        self.mamba_forw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                use_fast_path=False,
        )
        self.out_proj = copy.deepcopy(self.mamba_forw.out_proj)

        

        
        self.mamba_backw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                use_fast_path=False,

        )
        self.out_proj_backw = copy.deepcopy(self.mamba_backw.out_proj)


       
        self.mamba_forw.out_proj = nn.Identity()
        self.mamba_backw.out_proj = nn.Identity()
        self.x_Mamba = None
        self.y_Mamba = None

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        ll = len(x.shape)
     
       

        B, C = x.shape[:2]

        assert C == self.dim
   
        img_dims = x.shape[2:]



        if ll == 5: #3d
            
            Z,H,W = x.shape[2:]

            if Z%self.p==0 and H%self.p==0 and W%self.p==0:
                pool_layer = nn.AvgPool3d(self.p, stride=self.p)
                x_div = pool_layer(x)
            else:
                x_div = x

        elif ll == 4: #2d

            H,W = x.shape[2:]
            if self.p==0:
                self.p = 1
            if H%self.p==0 and W%self.p==0:
                pool_layer = nn.AvgPool2d(self.p, stride=self.p)
                x_div = pool_layer(x)
            else:
                x_div = x
        

      
        if ll == 5: #3d
            NZ,NH,NW = x_div.shape[2:]
        else:
            NH,NW = x_div.shape[2:]

        n_tokens = x_div.shape[2:].numel()
   

        x_flat = x_div.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        y_norm = torch.flip(x_norm, dims=[1])

        x_mamba = self.mamba_forw(x_norm)
        y_mamba = self.mamba_backw(y_norm)

        self.x_Mamba = self.out_proj(x_mamba).transpose(-1, -2)
        self.y_Mamba = self.out_proj_backw(y_mamba).transpose(-1, -2)

        x_out = self.out_proj(x_mamba + torch.flip(y_mamba, dims=[1]))

        if ll == 5:
            if Z%self.p==0 and H%self.p==0 and W%self.p==0:
                unpool_layer = nn.Upsample(scale_factor=self.p, mode='nearest')
                x_out = x_out.transpose(-1, -2).reshape(B, C, NZ, NH, NW)
                x_out = unpool_layer(x_out)
            else:
                x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)
        if ll == 4:
            if self.p==0:
                self.p = 1
            if H%self.p==0 and W%self.p==0:
                unpool_layer = nn.Upsample(scale_factor=self.p, mode='nearest')
                x_out = x_out.transpose(-1, -2).reshape(B, C, NH, NW)
                x_out = unpool_layer(x_out)
            else:
                x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)
                
        out = x_out + x

        return out
    
class HGMM(nn.Module):
    def __init__(self, depths=[1,1], 
                 dims=[64,64], **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        
        w_mamba_layers = []
        mamba_layers = []
        for i_layer in range(self.num_layers):
            for i_block in range(depths[i_layer]):
                input_channels = dims[i_layer]
                mamba_layers.append(BiPixelMambaLayer(input_channels, 2**( (self.num_layers-i_layer+1)//2-1) ))
                w_mamba_layers.append(BiWindowMambaLayerV2(input_channels, 2**((self.num_layers-i_layer+1)//2)//2 ))
            
        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.w_mamba_layers = nn.ModuleList(w_mamba_layers)

        w_mamba_layers_T = []
        mamba_layers_T = []
        for i_layer in range(self.num_layers):
            for i_block in range(depths[i_layer]):
                input_channels = dims[i_layer]
                mamba_layers_T.append(BiPixelMambaLayer(input_channels, 2**( (self.num_layers-i_layer+1)//2-1) ))
                w_mamba_layers_T.append(BiWindowMambaLayerV2(input_channels, 2**((self.num_layers-i_layer+1)//2)//2 ))
            
        self.mamba_layers_T = nn.ModuleList(mamba_layers_T)
        self.w_mamba_layers_T = nn.ModuleList(w_mamba_layers_T)
        self.GCN = GCN(dims[0]*4, dims[0]*2, dims[0])


    def forward(self,x):
        old = x #torch.Size([8, 64, 64, 64])
        lastx = x
        lastx_T = x.permute(0, 1, 3, 2)
        for s in range(self.num_layers):
            lastx_after = self.mamba_layers[s](lastx)
            lastx_after = self.w_mamba_layers[s](lastx_after)
            lastx_after = lastx + lastx_after
            lastx = lastx_after
        lastx_x_Mamba = self.w_mamba_layers[1].x_Mamba
        lastx_y_Mamba = self.w_mamba_layers[1].y_Mamba
        #print("lastx_x_Mamba {}".format(lastx_x_Mamba.shape))#torch.Size([8, 64, 4096])


        for s in range(self.num_layers):
            lastx_after_T = self.mamba_layers_T[s](lastx_T)
            lastx_after_T = self.w_mamba_layers_T[s](lastx_after_T)
            lastx_after_T = lastx_T + lastx_after_T
            lastx_T = lastx_after_T
        lastx_x_Mamba_T = self.w_mamba_layers_T[1].x_Mamba
        lastx_y_Mamba_T = self.w_mamba_layers_T[1].y_Mamba

        B, C, H, W = x.shape
        node_features = torch.cat([lastx_x_Mamba, lastx_y_Mamba, lastx_x_Mamba_T, lastx_y_Mamba_T], dim=1)
        node_features = node_features.permute(0,2,1)

        B1, M, F = node_features.shape
        adj_matrix = np.zeros((M, M))
        step = 32
        adj_matrix[::step, ::step] = 1 
        np.fill_diagonal(adj_matrix, 1)
        edge_index, edge_weight = dense_to_sparse(torch.tensor(adj_matrix))
        edge_index = edge_index.to(lastx_x_Mamba.device)


        gcn_x = self.GCN(node_features, edge_index)
        gcn_x = gcn_x.permute(0,2,1)
        gcn_x = gcn_x.reshape(B, C, H, W)

        return gcn_x + old + lastx + lastx_T.permute(0, 1, 3, 2)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu1(x)
        x = self.conv2(x, edge_index)
        return x

class BMD(nn.Module):
    def __init__(self):
        super().__init__()
        stages = []
        deconvs = []
        
        self.FSEM_1 = FSEMV2(128,32)
        self.FSEM_2 = FSEMV2(64,32)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        w_mamba_layers = []
        mamba_layers = []
        for i_layer in range(2):  
            mamba_layers.append(BiPixelMambaLayer(32, 2**( (2-i_layer+1)//2-1) ))
            w_mamba_layers.append(BiWindowMambaLayer(32, 2**((2-i_layer+1)//2)//2 ))

        stages.append(self.FSEM_1)
        stages.append(self.FSEM_2)
        deconvs.append(self.deconv1)
        deconvs.append(self.deconv2)
        self.stages = nn.ModuleList(stages)
        self.deconvs = nn.ModuleList(deconvs)
        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.w_mamba_layers = nn.ModuleList(w_mamba_layers)

        self.deconvall = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.out_pred = nn.Conv2d(16, 1, 1)

    def forward(self, skips, prediction):
        lres_input = skips[-1]
        x = self.stages[0](lres_input)
        x = torch.cat((x, prediction), 1)
        prediction = self.deconvs[0](x)
        prediction = self.mamba_layers[0](prediction)
        prediction = self.w_mamba_layers[0](prediction)

        lres_input = skips[-2]
        x = self.stages[1](lres_input)
        x = torch.cat((x, prediction), 1)
        prediction = self.deconvs[1](x)
        prediction = self.mamba_layers[1](prediction)
        prediction = self.w_mamba_layers[1](prediction)
        
        prediction = self.deconvall(prediction)
        prediction = self.out_pred(prediction)
        return prediction
