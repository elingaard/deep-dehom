import os
import math
from typing import Tuple

import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, Sampler
import torchvision.transforms.functional as TF
from collections import defaultdict
from itertools import repeat, chain

class GaussRBF:
    """Simple class representation of Gaussian radial basis function

    Args:
    c(float): center of kernel
    r(float): standard deviation

    """
    def __init__(self,c:float,r:float):
        self.c = c
        self.r = r

    def forward(self,x:np.ndarray)->np.ndarray:
        """Run input 'x' through the RBF kernel and return the corresponding activation 'h' """
        h = np.exp(-((x-self.c)**2)/self.r**2)
        return h

class RBFCollection:
    """Class for representing a collection of RBFs modeled using the 'GaussRBF' class

    Args:
    centers(np.array): array of RBF centers
    sigma(float): standard deviation used for all RBF kernels

    """
    def __init__(self,centers:np.ndarray,sigma:float):
        self.centers = centers
        self.sigma = sigma
        self.n_rbfs = len(centers)
        # initialize individual RBFs
        self.rbfs = []
        for c in self.centers:
            self.rbfs.append(GaussRBF(c,self.sigma))

    def forward(self,x:np.ndarray)->np.ndarray:
        """Forward passes an input 'x' through all RBFs in the collection"""
        self.H = np.zeros((self.n_rbfs,x.shape[0],x.shape[1]))
        # pass input through all rbfs
        for idx,rbf in enumerate(self.rbfs):
            self.H[idx,:,:] = rbf.forward(x)
        return self.H

    def index_sum(self,x:np.ndarray,sum_indices:np.ndarray)->np.ndarray:
        """Sum RBF activations based on an index vector. Used to model 2pi-periodicity
        and pi-symmetry, i.e. 45=-135 deg. Structure of the index vector should be an
        index for each RBF in the collection, i.e. if the activations from both RBF 2
        and 3 should be added to the RBF 1 activation the index vector would be: [1,1,1]
        """
        H = self.forward(x)
        nr_indices = len(np.unique(sum_indices))
        Hsum = np.zeros((nr_indices,x.shape[0],x.shape[1]))
        for i,sum_idx in enumerate(sum_indices):
            Hsum[sum_idx]+=H[i]
        return Hsum

    def activations2input(self,H:np.ndarray)->np.ndarray:
        """Convert the RBF activations from the 'index_sum' function to the corresponding input by
        using the inverse input-output mapping of the Gaussian:
        x = c +- sigma*sqrt(log(1/h))

        Highly looped function, not recommended for tasks where performance is key.

        Args:
        H(n_bins x H x W float matrix): RBF activations from the 'index_sum' function

        Returns:
        input_field(H x W float matrix): input corresponding to the RBF activations
        """
        # determine shape and preallocate reconstructed input field
        n_rbf_bins,height,width = H.shape
        input_field_recon = np.zeros((height,width))
        # loop over all activations pixel-wise
        for i in range(height):
            for j in range(width):
                # get activations for all bins at the pixel location
                hvec = H[:,i,j]
                # determine the two bins with the largest activation
                max_act_idxs = np.argsort(hvec)[-2:]
                # determine whether both bins are at the extreme of the input spectrum
                extrema_idx = [idx for idx in max_act_idxs if idx==0 or idx==(n_rbf_bins-1)]
                # special treatment if both bins are at the extrema
                if len(extrema_idx)==2:
                    # get the bin with maximum activation
                    idx = max_act_idxs[-1]
                    c = self.centers[idx]
                    # based on whether extrema is a one or the other end of the input spectrum either + or -
                    if idx==0:
                        x = c-self.sigma*np.sqrt(np.log(1/hvec[idx]))
                    elif idx==(n_rbf_bins-1):
                        x = c+self.sigma*np.sqrt(np.log(1/hvec[idx]))
                # if not extrema then determine the correct input by comparing the outputs from the two bins with
                # the highest activation. Inputs cannot be determined from just a single bin as the Gaussian is symmetric.
                else:
                    # create list for storing candidate inputs
                    x_cands = []
                    # loop both bins and evaluate +- values
                    for idx in max_act_idxs:
                        c = self.centers[idx]
                        x1 = c+self.sigma*np.sqrt(np.log(1/hvec[idx]))
                        x2 = c-self.sigma*np.sqrt(np.log(1/hvec[idx]))
                        x_cands+=[x1,x2]
                    x_cands = np.array(x_cands)
                    # get the unique candidates
                    uniq_cands,uniq_cnts = np.unique(np.round(x_cands,decimals=3),return_counts=True)
                    if len(uniq_cands)==4:
                        print("Candidate list:",uniq_cands)
                        raise Exception("""There cannot be 4 unique values at the same location. Please inspect the unique
                        candidates to determine rounding errors caused this error.""")
                    # if two candidates are equal then that input must be the correct one
                    x = uniq_cands[uniq_cnts>=2] # >=2 as input values very near centers might otherwise cause trouble
                # store pixel-wise input in global input field
                input_field_recon[i,j] = x

        return input_field_recon

class GroupedBatchSampler(BatchSampler):
    """
    Reference: https://github.com/pytorch/vision/blob/master/references/detection/group_by_aspect_ratio.py

    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
    """
    def __init__(self, sampler, group_ids, batch_size):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = batch_size

    def __iter__(self):
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)

        num_batches = 0
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]
                num_batches += 1
                del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size

        # now we have run out of elements that satisfy
        # the group criteria, let's return the remaining
        # elements so that the size of the sampler is
        # deterministic
        expected_num_batches = len(self)
        num_remaining = expected_num_batches - num_batches
        if num_remaining > 0:
            # for the remaining batches, take first the buffers with largest number
            # of elements
            for group_id, _ in sorted(buffer_per_group.items(),
                                      key=lambda x: len(x[1]), reverse=True):
                remaining = self.batch_size - len(buffer_per_group[group_id])
                samples_from_group_id = _repeat_to_at_least(samples_per_group[group_id], remaining)
                buffer_per_group[group_id].extend(samples_from_group_id[:remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size
                yield buffer_per_group[group_id]
                num_remaining -= 1
                if num_remaining == 0:
                    break
        assert num_remaining == 0

    def __len__(self):
        return len(self.sampler) // self.batch_size

def _repeat_to_at_least(iterable, n):
    """Utility function used in the GroupedBatchSampler"""
    repeat_times = math.ceil(n / len(iterable))
    repeated = chain.from_iterable(repeat(iterable, repeat_times))
    return list(repeated)

def collate_fn(batch):
    return tuple(zip(*batch))

def get_group_ids(datapath):
    """Loop through each sample in the specified directory and determine the group ID based on
    the aspect ratio"""
    filenames = os.listdir(datapath)
    AR_list = []
    for fname in filenames:
        E_arr = np.load(os.path.join(datapath,fname))
        h,w = E_arr[0].shape
        AR_list.append(h/w)
    # convert each unique aspect-ratio to a group id
    enum_dict = {v: k for k, v in enumerate(set(AR_list))}
    group_ids = [enum_dict[n] for n in AR_list]
    return np.array(group_ids)

class HomogenizationData(torch.utils.data.Dataset):
    """Dataset class used to handle homogenization data. Loads homogenization data from .npy files
    and applies RBF-encoding and optional data augmentation (horizontal and vertical flip). A value can be specified
    to add Gaussian noise to the input used to create the RBF-encoding.

    Args:
    datapath(str): string indicating the path to the .npy files
    n_rbf_bins(int): number of bins the angle interval [0;pi] is split into
    std_factor(float): factor multiplied to the angular step value to calculate the RBF standard deviation: sigma_rbf = std_factor*angle_step
    noise_mag(float): magnitude of gaussian noise, if 'None' no noise is added
    """
    def __init__(self,datapath:str,n_rbf_bins:int=24,std_factor:float=2.0,noise_mag:float=None,hvflip=True):
        # check inputs
        if noise_mag is not None:
            assert noise_mag<np.deg2rad(2), "A noise magnitude of more than 2deg is unadvised"
        self.datapath = datapath
        filenames = os.listdir(datapath)
        self.filenames = [f for f in filenames if f.endswith(".npy")]
        if len(filenames)==0:
            raise RuntimeError("No '.npy' files in directory")
        self.n_rbf_bins = n_rbf_bins
        self.std_factor = std_factor
        self.noise_mag = noise_mag
        self.hvflip = hvflip
        # initialize RBF centers and std dev.
        angle_step = np.pi/n_rbf_bins
        sigma = std_factor*angle_step
        THETAs = np.arange(-np.pi,np.pi+angle_step,angle_step)
        # add boundary support RBFs up to 3 std. dev's from the extremas of THETAs
        n_supps = int(np.ceil((3*sigma)/angle_step))
        lhs_supp = [THETAs[0]-i*angle_step for i in range(1,n_supps+1)]
        rhs_supp = [THETAs[-1]+i*angle_step for i in range(1,n_supps+1)]
        lhs_supp_idx = [n_rbf_bins-i for i in range(1,n_supps+1)]
        rhs_supp_idx = [i for i in range(1,n_supps+1)]
        THETAs = np.hstack([THETAs,lhs_supp,rhs_supp])
        # create indices needed to perform a 2-pi periodic and pi-symmetric sum
        self.sum_indices = np.hstack([np.arange(0,n_rbf_bins),np.arange(0,n_rbf_bins),np.array([0]),lhs_supp_idx,rhs_supp_idx])
        self.rbf_field = RBFCollection(THETAs,sigma)

    def __len__(self):
        "Mandatory function which returns the length of the dataset"
        return len(self.filenames)

    def __getitem__(self, idx)->Tuple[torch.FloatTensor,torch.FloatTensor,torch.FloatTensor]:
        "Mandatory function which returns a single sample from the dataset"
        # load orientation vectors
        Evec = np.load(os.path.join(self.datapath,self.filenames[idx]))
        ex = Evec[0].astype(np.float32)
        ey = Evec[1].astype(np.float32)
        # apply RBF-encoding to orientation vectors
        H_rbf = self.rbf_encoding(ex,ey).astype(np.float32)
        # apply data augmentation
        H_rbf,ex,ey = self.transform(H_rbf,ex,ey)
        return (H_rbf,ex,ey)

    def rbf_encoding(self,ex:np.ndarray,ey:np.ndarray)->np.ndarray:
        """Function used to apply RBF encoding"""
        # convert unit-vectors to angles in the interval [-pi;pi]
        angular_field = np.arctan2(ey,ex)
        # add gauss noise if flagged
        if self.noise_mag is not None:
            angular_field+=np.random.normal(0,self.noise_mag,angular_field.shape)
        # calculate index summed RBF fields
        H = self.rbf_field.index_sum(angular_field,self.sum_indices)
        return H

    def transform(self,H_rbf:np.ndarray,ex:np.ndarray,ey:np.ndarray)->Tuple[torch.FloatTensor,torch.FloatTensor,torch.FloatTensor]:
        """Transform RBF encoded input and corresponding orientation vectors
        to tensors. If 'hvflip=True' also apply horizontal and vertical flips"""
        assert len(H_rbf.shape)==3
        assert len(ex.shape)==2
        assert len(ey.shape)==2

        H_rbf = torch.from_numpy(H_rbf) # 'from_numpy' does not expand dimension by 1
        ex = TF.to_tensor(ex)
        ey = TF.to_tensor(ey)
        # apply data augmentation
        if self.hvflip is True:
            # Random horizontal flipping
            if np.random.random() > 0.5:
                ex = -TF.hflip(ex)
                ey = TF.hflip(ey)
                H_rbf = torch.flip(H_rbf,dims=[0,2])
                H_rbf = torch.roll(H_rbf,1,dims=0)

            # Random vertical flipping
            if np.random.random() > 0.5:
                ex = TF.vflip(ex)
                ey = -TF.vflip(ey)
                H_rbf = torch.flip(H_rbf,dims=[0,1])
                H_rbf = torch.roll(H_rbf,1,dims=0)

        return H_rbf,ex,ey

def batch_rot90(H_rbf:torch.FloatTensor,ex:torch.FloatTensor,ey:torch.FloatTensor)->Tuple[torch.FloatTensor,torch.FloatTensor,torch.FloatTensor]:
    """Apply rot90 data augmentation batch-wise to satisfy the GroupedBatchSampler,
    i.e. all samples in the batches should have same aspect ratio and thus it cannot
    be applied individually to each sample in the batch"""
    # input must be 4-dimensional otherwise rotate will not act as expected
    assert len(H_rbf.size())==4
    assert len(ex.size())==4
    assert len(ey.size())==4
    ex_temp = ex.clone()
    ex = -torch.rot90(ey,dims=[2,3])
    ey = torch.rot90(ex_temp,dims=[2,3])
    H_rbf = torch.rot90(H_rbf,dims=[2,3])
    shift90 = H_rbf.shape[1]//2
    H_rbf = torch.roll(H_rbf,shift90,dims=1)
    return H_rbf,ex,ey
