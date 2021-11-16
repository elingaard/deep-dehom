import os
from typing import Tuple, List

import numpy as np
from fire import Fire
import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torchvision.utils import make_grid
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from models import DehomNet, get_net_dict
from dataset import HomogenizationData, batch_rot90

class DotLoss(nn.Module):
    """Dot-product loss between the gradients of an image and a set of specified unit-vectors.
    Image gradients are calculated in a convolutional manner using the Sobel operator.

    L(x,y) = |dI(x)/dx*ex + dI(y)/dy*ey|

    """
    def __init__(self):
        super(DotLoss, self).__init__()
        # define sobel kernels
        sobel_kernel = torch.tensor([[1.,0.,-1.],[2.,0.,-2.],[1.,0.,-1.]])
        self.sobel_x = sobel_kernel.view(1,1,3,3)
        self.sobel_y = sobel_kernel.T.view(1,1,3,3)

    def forward(self,img:torch.FloatTensor,ex:torch.FloatTensor,ey:torch.FloatTensor)->Tuple[torch.FloatTensor,torch.FloatTensor,
        Tuple[torch.FloatTensor,torch.FloatTensor]]:
        assert len(img.size())==4
        assert len(ex.size())==4
        assert len(ey.size())==4
        # apply sobel kernels to input image to get image gradients
        dImgDx = F.conv2d(img, self.sobel_x.to(img.device),padding=1)
        dImgDy = -F.conv2d(img, self.sobel_y.to(img.device),padding=1)
        # use average filter to avoid very small gradients causing NaN
        dImgDx = F.avg_pool2d(F.pad(dImgDx,pad=(1,1,1,1),mode='replicate'),kernel_size=3,stride=1)
        dImgDy = F.avg_pool2d(F.pad(dImgDy,pad=(1,1,1,1),mode='replicate'),kernel_size=3,stride=1)
        # normalize gradients to unit vectors
        norm_mat_img = torch.norm(torch.cat([dImgDx,dImgDy],axis=1),dim=1).unsqueeze(1)
        dImgDx_norm = dImgDx/norm_mat_img
        dImgDy_norm = dImgDy/norm_mat_img
        #img_grad_norm_min = norm_mat_img.min() # track minimum norm
        # upsample orientations to gradient resolution
        ex = F.interpolate(ex,size=dImgDx.shape[-2:],mode='bilinear')
        ey = F.interpolate(ey,size=dImgDy.shape[-2:],mode='bilinear')
        # normalize orientation vectors after upsamling
        norm_mat_evec = torch.norm(torch.cat([ex,ey],axis=1),dim=1).unsqueeze(1)
        ex = ex/norm_mat_evec
        ey = ey/norm_mat_evec
        # calculate dot-product loss
        img_dot_loss = torch.abs(dImgDx_norm*ex + dImgDy_norm*ey)
        # calculate mean value for each sample in the batch
        dot_loss = img_dot_loss.view(img_dot_loss.shape[0], -1).mean(1, keepdim=True)
        return dot_loss, img_dot_loss, (dImgDx,dImgDy)

class ImgFreqLoss(nn.Module):
    def __init__(self,img_shape:tuple,target_period:float,band_width:int=2):
        super(ImgFreqLoss, self).__init__()
        assert img_shape[-1]==img_shape[-2]
        self.img_shape = img_shape
        self.target_period = target_period
        self.band_width = band_width
        self.in_freq_band = self.get_freq_band_matrix()
        self.hamm_win_2d = self.create_2d_hamming_window()

    def create_2d_hamming_window(self)->torch.FloatTensor:
        B,C,H,W = self.img_shape
        hamm_x = torch.hamming_window(H)
        hamm_y = torch.hamming_window(W)
        hamm_win_2d = torch.outer(hamm_x,hamm_y).unsqueeze(0).unsqueeze(0)
        return hamm_win_2d

    def get_freq_band_matrix(self)->torch.BoolTensor:
        """Get boolean matrix indicating the frequency band in the 2D FFT image"""
        B,C,H,W = self.img_shape
        target_period_idx = H//self.target_period
        freq_range = fft.fftfreq(H,d=1).numpy()
        inner_period_idx = target_period_idx-self.band_width
        outer_period_idx = target_period_idx+self.band_width
        print("Freq. band:",1/freq_range[outer_period_idx],"-",1/freq_range[inner_period_idx],"pixels/period")
        img_center = (H//2,W//2)
        img_x,img_y = torch.meshgrid(torch.arange(H),torch.arange(W))
        inner_freq_band = ((img_x-img_center[0])**2+(img_y-img_center[1])**2)>inner_period_idx**2
        outer_freq_band = ((img_x-img_center[0])**2+(img_y-img_center[1])**2)<outer_period_idx**2
        in_freq_band = torch.logical_and(inner_freq_band,outer_freq_band)
        return in_freq_band

    def forward(self,img:torch.FloatTensor)->Tuple[float,torch.FloatTensor]:
        """Calculate the ratio between the total energy in the FFT, and the energy inside the
        desired frequency band"""
        B,C,H,W = self.img_shape
        img_fft = fft.fft2(img*self.hamm_win_2d.to(img.device))
        img_fft = fft.fftshift(img_fft)
        img_fft_mag = torch.abs(img_fft)
        total_energy = torch.sum(img_fft_mag**2,dim=(1,2,3))/(H*W)
        freq_band_mag = img_fft_mag[:,:,self.in_freq_band]
        freq_band_energy = torch.sum(freq_band_mag**2,dim=(1,2))/(H*W)
        energy_ratio = freq_band_energy/total_energy
        return energy_ratio, img_fft

def batch_tv_loss(img:torch.FloatTensor)->float:
    """Total variation loss"""
    assert len(img.size())==4
    b,ch,h,w = img.shape
    tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum(dim=(1,2,3))
    tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum(dim=(1,2,3))
    tv_batch_mean = (tv_h+tv_w)/(ch*h*w)
    return tv_batch_mean

class HomogenizationDataModule(pl.LightningDataModule):
    def __init__(self, datapath:str, batch_size:int=16, split_ratios:List[float]=[0.8,0.2],
                 n_rbf_bins:int=24,std_dev_factor:float=2.0,noise_mag:float=np.deg2rad(1)):
        super().__init__()
        self.datapath = datapath
        self.batch_size = batch_size
        self.split_ratios = split_ratios
        self.n_rbf_bins = n_rbf_bins
        self.std_dev_factor = std_dev_factor
        self.noise_mag = noise_mag
        
    def setup(self, stage=None)->None:
        train_datapath = os.path.join(self.datapath,"train")
        self.dset = HomogenizationData(datapath=train_datapath,n_rbf_bins=self.n_rbf_bins,
                                       std_factor=self.std_dev_factor,noise_mag=self.noise_mag,hvflip=True)

        train_size = round(len(self.dset)*self.split_ratios[0])
        val_size = round(len(self.dset)*self.split_ratios[1])
        train_dset, val_dset = random_split(self.dset,(train_size,val_size),
                                                       generator=torch.Generator().manual_seed(42))
        self.train_dset = train_dset
        self.val_dset = val_dset

        test_datapath = os.path.join(self.datapath,"test")
        self.test_dset = HomogenizationData(datapath=test_datapath,n_rbf_bins=self.n_rbf_bins,
                                            std_factor=self.std_dev_factor,noise_mag=self.noise_mag,hvflip=False)
    
    def train_dataloader(self):
        train_loader = DataLoader(self.train_dset, batch_size=self.batch_size, persistent_workers=True,
                                  num_workers=os.cpu_count()//2,pin_memory=False,shuffle=True,drop_last=True)
        return train_loader
    
    def val_dataloader(self):
        val_loader = DataLoader(self.val_dset, batch_size=self.batch_size*2, persistent_workers=True,
                                num_workers=os.cpu_count()//2,pin_memory=False,shuffle=False)
        return val_loader

    def test_dataloader(self):
        train_loader = DataLoader(self.test_dset, batch_size=self.batch_size*4, persistent_workers=True,
                                  num_workers=os.cpu_count()//2,pin_memory=False,shuffle=False)
        return train_loader

class DeepDehom(pl.LightningModule):
    def __init__(self,input_chs:int,out_dims:Tuple[int,int,int,int],lr:float=2e-4,lambda_orient:float=1.0,
                 lambda_tv:float=1.0, lambda_freq:float=1.0,lambda_fork:float=0.0, period:int=20,
                 model_size:str='medium', out_kernel_size:int=7):
        super().__init__()
        self.learning_rate = lr
        self.lambda_orient = lambda_orient
        self.lambda_tv = lambda_tv
        self.lambda_freq = lambda_freq
        self.lambda_fork = lambda_fork
        self.periodicity = period
        self.target_tv = (2*(1/self.periodicity))**2
        self.save_hyperparameters()
        net_dict = get_net_dict(input_dim=input_chs,model_size=model_size)
        self.model = DehomNet(net_dict,out_kernel_size=out_kernel_size).to(self.device)
        self.dot_loss = DotLoss()
        self.mse_loss = nn.MSELoss()
        self.img_freq_loss = ImgFreqLoss(out_dims,target_period=self.periodicity,band_width=3)

    def forward(self,x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        H_rbf,ex,ey = batch
        if np.random.random()>0.5:
            H_rbf,ex,ey = batch_rot90(H_rbf,ex,ey)
        
        # generate rho
        rho_tilde = self(H_rbf)
        # dot-product loss
        dot_loss_val,img_dot,img_grads = self.dot_loss(rho_tilde,ex,ey)
        orient_loss = self.lambda_orient*torch.mean(dot_loss_val)
        # forking loss
        img_dot_indicator = img_dot*(1-rho_tilde)
        fork_loss = self.lambda_fork*torch.mean(img_dot_indicator)
        # total variation loss
        tv_loss_val = batch_tv_loss(rho_tilde)/self.target_tv
        tv_mse_target = torch.ones(tv_loss_val.shape).to(self.device)
        tv_loss = self.lambda_tv*self.mse_loss(tv_loss_val,tv_mse_target)
        # freq loss
        freq_loss_val,rho_tilde_fft = self.img_freq_loss(rho_tilde)
        freq_loss = -self.lambda_freq*torch.mean(freq_loss_val)
        # total loss
        total_loss = orient_loss + tv_loss + fork_loss + freq_loss

        loss_dict = {"train_orient_loss":orient_loss,"train_tv_loss":tv_loss,
                     "train_fork_loss":fork_loss,"train_freq_loss":freq_loss}
        self.log_dict(loss_dict)

        if self.global_step%1==0:
            img_grid = make_grid(rho_tilde[:8],nrow=4)
            self.logger.experiment.add_image('density fields', img_grid, self.global_step)

        return total_loss

    def validation_step(self, batch, batch_idx):
        H_rbf,ex,ey = batch
        # generate rho
        rho_tilde = self(H_rbf)
        # dot-product loss
        dot_loss_val,img_dot,img_grads = self.dot_loss(rho_tilde,ex,ey)
        orient_loss = self.lambda_orient*torch.mean(dot_loss_val)
        # forking loss
        img_dot_indicator = img_dot*(1-rho_tilde)
        fork_loss = self.lambda_fork*torch.mean(img_dot_indicator)
        # total variation loss
        tv_loss_val = batch_tv_loss(rho_tilde)/self.target_tv
        tv_mse_target = torch.ones(tv_loss_val.shape).to(self.device)
        tv_loss = self.lambda_tv*self.mse_loss(tv_loss_val,tv_mse_target)
        # freq loss
        freq_loss_val,rho_tilde_fft = self.img_freq_loss(rho_tilde)
        freq_loss = -self.lambda_freq*torch.mean(freq_loss_val)
        # total loss
        total_loss = orient_loss + tv_loss + fork_loss + freq_loss

        loss_dict = {"val_orient_loss":orient_loss,"val_tv_loss":tv_loss,
                     "val_fork_loss":fork_loss,"val_freq_loss":freq_loss}
        self.log_dict(loss_dict)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

def run_trainer(datapath="../synthethic_orientation_fields", batch_size:int=32, model_size:str="medium", 
                lambda_orient:float=1.0, lambda_tv:float=1.0, lambda_fork:float=0.0, lambda_freq:float=1.0, 
                period:int=20, n_rbf_bins:int=24,std_dev_factor:float=2.0):

    dm = HomogenizationDataModule(datapath,batch_size,n_rbf_bins=n_rbf_bins,std_dev_factor=std_dev_factor)
    dm.setup()
    inp_field_size = (80,80) # must be square
    out_field_dims = (batch_size,1,4*inp_field_size[0],4*inp_field_size[1])
    deep_dehom = DeepDehom(input_chs=n_rbf_bins,out_dims=out_field_dims, lambda_orient=lambda_orient,
                           lambda_tv=lambda_tv,lambda_fork=lambda_fork,lambda_freq=lambda_freq,
                           period=period,model_size=model_size,out_kernel_size=7)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        verbose=True,
        mode='min',
    )

    trainer = pl.Trainer(gpus=0,max_epochs=50,check_val_every_n_epoch=1,progress_bar_refresh_rate=1,log_every_n_steps=1,
                         callbacks=[checkpoint_callback,early_stop_callback])
    trainer.fit(deep_dehom,dm)

if __name__=='__main__':
    Fire(run_trainer)
