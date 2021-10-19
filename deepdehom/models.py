import torch.nn as nn
import torch.nn.functional as F
from more_itertools import pairwise

class ConvBatchReLU(nn.Module):
    """Convience class for performing Conv2d + BatchNorm2d + ReLU"""
    def __init__(self, n_in, n_out, kernel_size, stride=1, bias=True, dilation=1, groups=1, padding='same',alpha=0):
        super(ConvBatchReLU, self).__init__()
        if padding=='same':
            self.npad = (kernel_size+(kernel_size-1)*(dilation-1)-1)//2
        else:
            self.npad = padding

        self.conv = nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, bias=bias, dilation=dilation, groups=groups,padding=self.npad),
            nn.BatchNorm2d(n_out),
            nn.LeakyReLU(negative_slope=alpha),
        )
    def forward(self, x):
        out = self.conv(x)
        return out

class ResNetBlock(nn.Module):
    """ Class for creating a residual network block, where the network
    predicts the residual between the input x and output y:
    y = f(x) + x -> f(x) = y - x
    """
    def __init__(self, n_channels,kernel_size,bias=True,dilation=1):
        super(ResNetBlock, self).__init__()
        self.npad = (kernel_size+(kernel_size-1)*(dilation-1)-1)//2
        self.residual = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, bias=bias, dilation=dilation, stride=1,padding=self.npad),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, bias=bias, dilation=dilation, stride=1,padding=self.npad),
        )
    def forward(self, x):
        res = self.residual(x)
        out = F.relu(x+res)
        return out

class DehomNet(nn.Module):
    def __init__(self,net_dict,out_kernel_size=7):
        super().__init__()
        # define model structure
        n_chs_in = net_dict["n_ch_in"]
        n_chs_res = net_dict["n_ch_res"]
        n_blocks_res = net_dict["n_blocks_res"]
        n_chs_up = net_dict["n_chs_up"]
        n_chs_out = net_dict["n_chs_out"]
        # check network dimensions are compatible
        assert n_chs_res == n_chs_up[0]
        out_pad = (out_kernel_size-1)//2

        #initialize layers
        self.input_layer = nn.Sequential(
                     ConvBatchReLU(n_chs_in,n_chs_res//2,kernel_size=7),
                     ConvBatchReLU(n_chs_res//2,n_chs_res,kernel_size=5)
                    )
        self.resnet = nn.Sequential(*[ResNetBlock(n_chs_res,kernel_size=3) for _ in range(n_blocks_res)])
        up_layers = []
        for (n_ch1,n_ch2) in pairwise([n_chs_res]+n_chs_up):
            up_layers.append(nn.Sequential(
                         nn.Upsample(scale_factor=2,mode='nearest'),
                         ConvBatchReLU(n_ch1,n_ch1,kernel_size=3),
                         ConvBatchReLU(n_ch1,n_ch2,kernel_size=3),
                        ))
        self.upsampling = nn.Sequential(*up_layers)
        self.output_layer = nn.Sequential(
            nn.Conv2d(n_chs_up[-1],n_chs_out, kernel_size=out_kernel_size,padding=out_pad),
            nn.Sigmoid()
        )
    def forward(self,x):
        x_in = self.input_layer(x)
        x_res = self.resnet(x_in)
        x_up = self.upsampling(x_res)
        y = self.output_layer(x_up)
        return y

def get_net_dict(input_dim:int,model_size="medium")->dict:
    if model_size=="small":
        net_dict = {}
        net_dict["n_ch_in"] = input_dim
        net_dict["n_ch_res"] = 32
        net_dict["n_blocks_res"] = 2
        net_dict["n_chs_up"] = [32,16,16]
        net_dict["n_chs_out"] = 1
    elif model_size=="medium":
        net_dict = {}
        net_dict["n_ch_in"] = input_dim
        net_dict["n_ch_res"] = 64
        net_dict["n_blocks_res"] = 4
        net_dict["n_chs_up"] = [64,32,32]
        net_dict["n_chs_out"] = 1
    elif model_size=="large":
        net_dict = {}
        net_dict["n_ch_in"] = input_dim
        net_dict["n_ch_res"] = 64
        net_dict["n_blocks_res"] = 8
        net_dict["n_chs_up"] = [64,32,32]
        net_dict["n_chs_out"] = 1
    elif model_size=="medium_up2":
        net_dict = {}
        net_dict["n_ch_in"] = input_dim
        net_dict["n_ch_res"] = 64
        net_dict["n_blocks_res"] = 4
        net_dict["n_chs_up"] = [64,32]
        net_dict["n_chs_out"] = 1

    return net_dict