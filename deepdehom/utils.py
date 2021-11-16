from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology as morph
import skimage.transform as transform
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt
from scipy.io import loadmat

def load_homogenization_file(filepath:str)->Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,float]:
    """Load Matlab file containing the homogenization results"""
    mat_dict = loadmat(filepath)
    mat_filename = filepath.split("/")[-1]
    nely,nelx = mat_dict["nDim"][0]
    nely = nely.astype(int)
    nelx = nelx.astype(int)
    Nvecs = mat_dict["N"]
    MUs = mat_dict["w"]
    comp = mat_dict["c"]
    mu1 = MUs[:,1].reshape((nelx,nely),order='F')
    mu2 = MUs[:,0].reshape((nelx,nely),order='F')
    N1x = Nvecs[:,0].reshape((nelx,nely),order='F')
    N1y = Nvecs[:,1].reshape((nelx,nely),order='F')
    N2x = Nvecs[:,2].reshape((nelx,nely),order='F')
    N2y = Nvecs[:,3].reshape((nelx,nely),order='F')
    if "clamCentSym" in mat_filename or "MBBSym" in mat_filename:
        N1x[:,nely//2:] = -N1x[:,nely//2:]
        N2x[:,nely//2:] = -N2x[:,nely//2:]
    return N1x,N1y,N2x,N2y,mu1,mu2,comp

def get_hist_periodicity(rho1_tilde:np.ndarray,rho2_tilde:np.ndarray)->float:
    """Get the periodicity of a field based on the histogram of the fields
    distance transform"""
    # get distance from from skeleton image of rho_tilde's
    skel1 = morph.skeletonize(rho1_tilde>rho1_tilde.mean())
    skel2 = morph.skeletonize(rho2_tilde>rho2_tilde.mean())
    dist1_trans = distance_transform_edt(~skel1)
    dist2_trans = distance_transform_edt(~skel2)
    # merge distance fields and calculate hisogram over distance values
    merged_dist_trans = np.concatenate((dist1_trans.flatten(),dist2_trans.flatten()))
    merged_dist_trans = np.round(merged_dist_trans).astype(int)
    dist_vals,dist_cnts = np.unique(merged_dist_trans,return_counts=True)
    avg_cnt = np.sum(dist_cnts)/len(dist_cnts)
    half_period = sum((dist_cnts/avg_cnt)>0.25)
    full_period = 2*half_period
    return full_period

def get_homogenization_volfrac(mu1:np.ndarray,mu2:np.ndarray)->float:
    """Get the volume fraction based on the lamination width fields"""
    ele_vol = mu1+mu2-mu1*mu2
    volfrac = np.sum(ele_vol)/ele_vol.size
    return volfrac

class LaminationWidthProjection:
    """
    Class used to project lamination width onto the intermediate density field
    utilizes several different image transformations, such as skeletonize and
    distance transform

    Properties:
    HMIN(int): lowest number of pixels used to resolve a structural element
    WMIN(float): minimum relative thickness
    NN_USR(int): upsampling rate of the neural network
    FIELD_USR(int): upsampling rate of the intermediate field if None the minimum required upsampling rate is used
    periodicity(int): periodicity of the intermediate field in pixels/period
    npad(int): padding used on neural network input (helps ensure a smoother skeletonize near edges)

    """
    def __init__(self,periodicity:int,hmin:int,wmin:float,nn_usr:int,field_usr:int=None,input_pad:int=0):
        self.periodicity = periodicity
        self.HMIN = hmin
        self.WMIN = wmin
        self.NN_USR = nn_usr
        min_field_usr = np.ceil(self.HMIN/(self.WMIN*self.periodicity))
        if field_usr is None:
            self.FIELD_USR = min_field_usr
        else:
            self.FIELD_USR = field_usr
        assert self.FIELD_USR>=min_field_usr, "Field upsampling rate must be high enough to resolve minimum line thickness"
        self.npad = int(input_pad*self.FIELD_USR*self.NN_USR)

    def solidify_branching_regions(self,rho_tilde:np.ndarray,dot_loss_surface:np.ndarray)->np.ndarray:
        """Uses the dot-product loss surface to create a solid circle around branching
        points in rho_tilde"""
        # smooth the dot-product loss surface
        smth_dot_loss_surf = gaussian_filter(dot_loss_surface,sigma=1.5)
        # set values near the edge to zero (branching points in this region are not interesting)
        n_edge = np.round(np.min(dot_loss_surface.shape)*0.02).astype(int)
        smth_dot_loss_surf[:,-n_edge:] = 0
        smth_dot_loss_surf[:,:n_edge] = 0
        smth_dot_loss_surf[-n_edge:,:] = 0
        smth_dot_loss_surf[:n_edge,:] = 0
        # find maxima in the dot-product loss surface
        binary_maxima_matrix = morph.h_maxima(smth_dot_loss_surf,h=smth_dot_loss_surf.max()*0.5)
        # dilate maxima matrix
        dilation_radius = self.periodicity//4
        dilated_maxima_matrix = morph.binary_dilation(binary_maxima_matrix,morph.disk(dilation_radius))
        rho_tilde[dilated_maxima_matrix] = 1
        return rho_tilde

    def get_dilated_skeleton(self,rho_tilde:np.ndarray)->np.ndarray:
        """
        Performs a skeletonize of the intermediate field and afterwards
        remove input padding and dilates the skeleton
        """
        skel = morph.skeletonize(rho_tilde>rho_tilde.mean())
        skel_nopad = skel[self.npad:-self.npad,self.npad:-self.npad]
        dilated_skel = morph.binary_dilation(skel_nopad,selem=morph.disk(1))
        return dilated_skel

    def plot_projection_fields(self,rho_tilde:np.ndarray,skel:np.ndarray,dist_trans:np.ndarray,
                               mu:np.ndarray,rho1_hr:np.ndarray,rho2_hr:np.ndarray):
        """
        Plots the relevant fields used to perform the lamination width projection
        """
        fig_dim = (18,12)
        fig,axarr = plt.subplots(3,2,figsize=(fig_dim))
        axarr[0,0].imshow(-rho_tilde[self.npad:-self.npad,self.npad:-self.npad],interpolation=None,cmap='gray')
        axarr[0,0].set_title("rho_tilde")
        axarr[0,1].imshow(-(skel.astype(int)),cmap='gray',interpolation=None)
        axarr[0,1].set_title("skeleton")
        axarr[1,0].imshow(dist_trans,cmap='jet',interpolation=None)
        axarr[1,0].set_title("clipped distance transform")
        axarr[1,1].imshow(mu,vmin=0,vmax=1,cmap='jet',interpolation=None)
        axarr[1,1].set_title("clipped lamination width")
        axarr[2,0].imshow(-rho1_hr,interpolation=None,cmap='gray')
        axarr[2,0].set_title("rho1_hr")
        axarr[2,1].imshow(-rho2_hr,cmap='gray',interpolation=None)
        axarr[2,1].set_title("rho2_hr")
        plt.tight_layout()
        plt.show()

    def forward(self,rho_tilde:Tuple[np.ndarray,np.ndarray],dot_loss_surf:Tuple[np.ndarray,np.ndarray],mu:Tuple[np.ndarray,np.ndarray],
                refine_period=False,wmin_thres=True,verbose=False,enable_plot=False)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """
        Forward pass of the lamination width projection scheme

        Args:
        rho_tilde(tuple of NxM arrays): intermediate field in both directions
        dot_loss_surf(tuple of NxM arrays): dot-product loss surface for each intermediate field
        mu(tuple of NxM arrays): lamination width field for each direction
        refine_period(bool): refine period using a histogram of the distance transform
        verbose(bool): print relevant output during projection
        enable_plot(bool): flag used to enable/disable plotting (default:False)
        wmin_thres(bool): whether to set all structural members with a thickness
                            lower than the minimum relative thickness to zero (default:True)

        Returns:
        rho_hr(NxM array): de-homogenized density field
        """

        # unpack input
        rho1_tilde,rho2_tilde = rho_tilde
        dot_loss_surf1, dot_loss_surf2 = dot_loss_surf
        mu1,mu2 = mu
        if verbose is True: print("Target volfrac:",get_homogenization_volfrac(mu1,mu2))

        # set regions around dot loss surface maxima to solid
        rho1_tilde = self.solidify_branching_regions(rho1_tilde,dot_loss_surf1)
        rho2_tilde = self.solidify_branching_regions(rho2_tilde,dot_loss_surf2)

        # upsample intermediate fields based on periodicity of the intermediate fields
        rho1_tilde_up = transform.rescale(rho1_tilde,scale=self.FIELD_USR,order=1,preserve_range=True)
        rho2_tilde_up = transform.rescale(rho2_tilde,scale=self.FIELD_USR,order=1,preserve_range=True)

        # skeletonize and dilate upsampled intermediate fields
        if verbose is True: print("Performing skeletonize...")
        skel1 = self.get_dilated_skeleton(rho1_tilde_up)
        skel2 = self.get_dilated_skeleton(rho2_tilde_up)

        # distance transform of upsampled skeleton
        if verbose is True: print("Performing distance transform...")
        dist1_trans = distance_transform_edt(~skel1)
        dist2_trans = distance_transform_edt(~skel2)

        # clip distance transform according to upsampled periodicity
        if refine_period==True:
            period_up = get_hist_periodicity(rho1_tilde_up,rho2_tilde_up)
            print("Refined period:",period_up)
        else:
            period_up = self.FIELD_USR*self.periodicity
        dist1_clipped = normalize_data(np.clip(dist1_trans,0,period_up/2))
        dist2_clipped = normalize_data(np.clip(dist2_trans,0,period_up/2))

        # upsample lamination widths to match upsampled rho
        mu1_up = transform.rescale(mu1,scale=self.FIELD_USR*self.NN_USR,order=1,preserve_range=True)
        mu2_up = transform.rescale(mu2,scale=self.FIELD_USR*self.NN_USR,order=1,preserve_range=True)

        # clip lamination width according to minimum relative width resolvable by the mesh
        WMIN_mesh = self.HMIN/(period_up)
        mu1_clipped = (np.clip(mu1_up,WMIN_mesh,1)-WMIN_mesh)/(1-WMIN_mesh)
        mu2_clipped = (np.clip(mu2_up,WMIN_mesh,1)-WMIN_mesh)/(1-WMIN_mesh)

        # project lamination width onto upsampled intermediate fields
        rho1_hr = ((mu1_clipped-dist1_clipped)>=0).astype(np.int16)
        rho2_hr = ((mu2_clipped-dist2_clipped)>=0).astype(np.int16)
        if wmin_thres is True:
            rho1_hr[mu1_up<self.WMIN] = 0
            rho2_hr[mu2_up<self.WMIN] = 0

        # perform of the density fields in each orientation
        rho_hr = np.minimum(rho1_hr+rho2_hr,np.ones(rho1_hr.shape,dtype=int))
        # clean up design by removing disconnected objects smaller than a certain size
        rho_hr = morph.remove_small_objects(rho_hr.astype(bool),min_size=512).astype(int)

        if verbose is True:
            rho_hr_volfrac = np.sum(rho_hr)/rho_hr.size
            print("Actual volume fraction:",rho_hr_volfrac)

        if enable_plot is True:
            self.plot_projection_fields(rho1_tilde_up,skel1,dist1_clipped,mu1_clipped,rho1_hr,rho2_hr)

        return rho_hr, rho1_hr, rho2_hr

def normalize_data(x:np.ndarray)->np.ndarray:
    x = (x-x.min())/(x.max()-x.min())
    return x