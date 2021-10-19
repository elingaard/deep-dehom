import argparse
import os
import io
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
from tqdm import tqdm
from fire import Fire
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import DehomNet, get_net_dict
from dataset import HomogenizationData, batch_rot90, collate_fn
from utils import DotLoss, ImgFreqLoss, batch_tv_loss, count_model_parameters, plot_img_with_colorbar

def tensorboard_plot(rho:torch.FloatTensor,img_dot_loss:torch.FloatTensor):
    rho_np = rho.detach().cpu().numpy().squeeze(1)
    img_dot_loss_np = img_dot_loss.detach().cpu().numpy().squeeze(1)

    # determine figure size based on aspect ratio
    h,w = rho_np.shape[-2:]
    AR = h//w
    if AR<0.5:
        plot_size = (16,4)
    elif AR<0.75:
        plot_size = (14,6)
    elif AR<1.5:
        plot_size = (12,8)
    elif AR<2.5:
        plot_size = (10,10)
    else:
        plot_size = (8,12)
    n_samples = 4
    fig,axarr = plt.subplots(2,4,figsize=plot_size)
    for idx in range(n_samples):
        axarr[0,idx].imshow(-rho_np[idx],cmap='gray')
        axarr[0,idx].axis("off")
        plot_img_with_colorbar(img_dot_loss_np[idx],axarr[1,idx])
        axarr[1,idx].axis("off")

    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

class DeepDehom:
    def __init__(self,DATA_PATH:str,pretrained:str="",batch_size:int=32,num_epochs:int=50, model_size:str="medium", out_kernel_size:int=7, 
                 lambda_orient:float=1.0, lambda_tv:float=1.0, lambda_fork:float=0.0, lambda_freq:float=1.0, period:int=20,
                 n_rbf_bins:int=24,std_dev_factor:float=2.0, MODEL_PREFIX:str='run0'):
        """Class for training and evaluating the deep dehomogenization network"""
        self.DATA_PATH = DATA_PATH
        format_arg_list = [num_epochs,batch_size,n_rbf_bins,std_dev_factor,period,lambda_orient,lambda_tv,lambda_fork,lambda_freq]
        model_name_str = MODEL_PREFIX + "_e{}_b{}_nb{}_std{}_p{}_ldot{}_:tv{}_lfk{}_lfq{}".format(*format_arg_list)
        self.MODEL_PATH = os.path.join("pretrained/DeepDehom",model_name_str)
        if os.path.exists(self.MODEL_PATH) is False:
            os.makedirs(self.MODEL_PATH)
            os.makedirs(os.path.join(self.MODEL_PATH,"weights"))

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lambda_orient = lambda_orient
        self.lambda_tv = lambda_tv
        self.lambda_fork = lambda_fork
        self.lambda_freq = lambda_freq
        self.periodicity = period
        self.target_tv = (2*(1/self.periodicity))**2
        #self.target_tv = 0.01
        self.n_rbf_bins = n_rbf_bins
        self.std_dev_factor = std_dev_factor
        self.pretrained = pretrained
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialize datasets
        train_datapath = os.path.join(self.DATA_PATH,"train")
        test_datapath = os.path.join(self.DATA_PATH,"test")
        print("Initializing datasets")
        self.train_dset = HomogenizationData(datapath=train_datapath,n_rbf_bins=self.n_rbf_bins,std_factor=self.std_dev_factor,noise_mag=np.deg2rad(1),hvflip=True)
        self.test_dset = HomogenizationData(datapath=test_datapath,n_rbf_bins=self.n_rbf_bins,std_factor=self.std_dev_factor,noise_mag=np.deg2rad(1),hvflip=False)
        self.train_loader = DataLoader(self.train_dset,num_workers=4,collate_fn=collate_fn,batch_size=self.batch_size,
                          pin_memory=(torch.cuda.is_available()),drop_last=True)
        self.test_loader = DataLoader(self.test_dset,num_workers=4,collate_fn=collate_fn,batch_size=self.batch_size,
                          pin_memory=(torch.cuda.is_available()),drop_last=True)
        self.test_iterator = iter(self.test_loader)

        self.dot_loss = DotLoss()
        self.mse_loss = nn.MSELoss()
        self.img_freq_loss = ImgFreqLoss((self.batch_size,1,320,320),target_period=self.periodicity,band_width=3)

        # initialize model
        net_dict = get_net_dict(input_dim=self.n_rbf_bins,model_size=model_size)
        self.model = DehomNet(net_dict,out_kernel_size=out_kernel_size).to(self.device)
        if len(self.pretrained)>0:
            self.model.load_state_dict(torch.load(self.pretrained,map_location=self.device),strict=True)
            print("Using pretrained model")

    def train(self):

        tv_mse_target = torch.ones(self.batch_size).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), 2e-4)
        print("Model parameters: ",count_model_parameters(self.model))

        summ_writer = SummaryWriter(log_dir=self.MODEL_PATH)
        it = 0
        for epoch in range(self.num_epochs):
            for batch_nr, data_tuple in enumerate(tqdm(self.train_loader)):
                optimizer.zero_grad()
                # run RBF encoded orientations through network
                H_rbf,ex,ey = data_tuple
                H_rbf = torch.stack(H_rbf).to(self.device)
                ex = torch.stack(ex).to(self.device)
                ey = torch.stack(ey).to(self.device)
                if np.random.random()>0.5:
                    H_rbf,ex,ey = batch_rot90(H_rbf,ex,ey)
                # generate rho
                rho_tilde = self.model(H_rbf)
                # dot-product loss
                dot_loss_val,img_dot,img_grads = self.dot_loss(rho_tilde,ex,ey)
                train_orient_loss = self.lambda_orient*torch.mean(dot_loss_val)
                # forking loss
                img_dot_indicator = img_dot*(1-rho_tilde)
                train_fork_loss = self.lambda_fork*torch.mean(img_dot_indicator)
                # total variation loss
                tv_loss_val = batch_tv_loss(rho_tilde)/self.target_tv
                train_tv_loss = self.lambda_tv*self.mse_loss(tv_loss_val,tv_mse_target)
                # freq loss
                freq_loss_val,rho_tilde_fft = self.img_freq_loss(rho_tilde)
                train_freq_loss = -self.lambda_freq*torch.mean(freq_loss_val)
                # total loss
                total_loss = train_orient_loss + train_tv_loss + train_fork_loss + train_freq_loss
                total_loss.backward()
                optimizer.step()

                # evaluate test loss
                test_rho_tilde, test_img_dot, test_orient_loss, test_tv_loss, test_fork_loss, test_freq_loss = self.eval()
                self.model.train()

                # write loss and network output to tensorboard
                it+=1
                summ_writer.add_scalars('train', {'tv':train_tv_loss.item(),
                                                  'orient':train_orient_loss.item(),
                                                  'fork':train_fork_loss.item(),
                                                  'freq':-train_freq_loss.item()},it)
                summ_writer.add_scalars('test', {'tv':test_tv_loss.item(),
                                                  'orient':test_orient_loss.item(),
                                                  'fork':test_fork_loss.item(),
                                                  'freq':-test_freq_loss.item()},it)
                # store image of generated designs every N iteration
                if batch_nr%100==0:
                    tb_image = PIL.Image.open(tensorboard_plot(test_rho_tilde,test_img_dot))
                    tb_image = transforms.ToTensor()(tb_image)
                    summ_writer.add_image('Gen. images', tb_image, it)

            print("Saving model at "+self.MODEL_PATH)
            torch.save(self.model.state_dict(), os.path.join(self.MODEL_PATH,"weights","epoch"+str(epoch)+".pth"))

    def eval(self)->Tuple[torch.FloatTensor,torch.FloatTensor,float,float,float,float]:
        # reinitialize test loader once its run out of elements
        try:
            data_tuple = next(self.test_iterator)
        except StopIteration:
            self.test_iterator = iter(self.test_loader)
            data_tuple = next(self.test_iterator)
        # calculate test loss
        self.model.eval()
        with torch.no_grad():
            # run RBF encoded orientations through network
            H_rbf,ex,ey = data_tuple
            H_rbf = torch.stack(H_rbf).to(self.device)
            ex = torch.stack(ex).to(self.device)
            ey = torch.stack(ey).to(self.device)
            rho_tilde = self.model(H_rbf)
            # dot-product loss
            dot_loss_val,img_dot, img_grads = self.dot_loss(rho_tilde,ex,ey)
            test_orient_loss = self.lambda_orient*torch.mean(dot_loss_val)
            # forking loss
            img_dot_indicator = img_dot*(1-rho_tilde)
            test_fork_loss = self.lambda_fork*torch.mean(img_dot_indicator)
            # total variation loss
            tv_mse_target = torch.ones(self.batch_size).to(self.device)
            tv_loss_val = batch_tv_loss(rho_tilde)/self.target_tv
            test_tv_loss = self.lambda_tv*self.mse_loss(tv_loss_val,tv_mse_target)
            # freq loss
            freq_loss_val,rho_tilde_fft = self.img_freq_loss(rho_tilde)
            test_freq_loss = -self.lambda_freq*torch.mean(freq_loss_val)

        return rho_tilde,img_dot,test_orient_loss,test_tv_loss,test_fork_loss,test_freq_loss

def train_model(DATA_PATH:str,pretrained:str="",batch_size:int=32,num_epochs:int=50, model_size:str="medium", out_kernel_size:int=7, 
                 lambda_orient:float=1.0, lambda_tv:float=1.0, lambda_fork:float=0.0, lambda_freq:float=1.0, period:int=20,
                 n_rbf_bins:int=24,std_dev_factor:float=2.0, MODEL_PREFIX:str='run0'):

    deep_dehom = DeepDehom(DATA_PATH,pretrained,batch_size,num_epochs, model_size, out_kernel_size, 
                           lambda_orient, lambda_tv, lambda_fork, lambda_freq, period, n_rbf_bins,std_dev_factor, MODEL_PREFIX)
    deep_dehom.train()

if __name__ == "__main__":
    Fire(train_model)




