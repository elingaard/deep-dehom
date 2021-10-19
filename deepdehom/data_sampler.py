import os
from typing import Tuple

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RectBivariateSpline
from skimage import measure
from tqdm import tqdm
from fire import Fire

class FourierFieldSampler:
    """Class used to sample a random patch from a fourier field.

    Fourier field is created using only the sine-sine contribution of the Fourier expansion:
    f(x,y) = sum_{n,m}[ c_{n,m}*sin(n*pi*x/xL)*sin(m*pi*y/yL) ]

    Args:
    xL(float): length of domain in x-direction
    yL(float): length of domain in y-direction
    Ngrid(int): Number of points in each direction of the square domain
    n_coeffs(int): Number of Fourier coefficients (equal in each direction)

    """
    def __init__(self,xL:float,yL:float,Ngrid:int,n_coeffs:int):
        self.xL = xL
        self.yL = yL
        self.Ngrid = Ngrid
        self.n_coeffs = n_coeffs

    def gen_rand_field(self):
        """Generate a random fourier field and stores the field and its derivatives in the class (F,dFdx,dFdy)"""
        N = M = self.n_coeffs
        x,y = np.meshgrid(np.linspace(0,self.xL,self.Ngrid),np.linspace(0,self.yL,self.Ngrid))
        coeffs = np.random.standard_normal(size=(N,N))
        # put more emphasis on low-frequencies
        wN = np.linspace(1,0.25,N).reshape(N,1)
        wM = np.linspace(1,0.25,M).reshape(M,1)
        Wmat = np.matmul(wN,wM.T)
        coeffs*=Wmat
        F = 0
        dFdx = 0
        dFdy = 0
        for n in range(N):
            for m in range(M):
                F += coeffs[n,m]*np.sin(n*np.pi*x/self.xL)*np.sin(m*np.pi*y/self.yL)
                # analytical gradients
                dFdx += (np.pi*n*coeffs[n,m]*np.cos(n*np.pi*x/self.xL)*np.sin(m*np.pi*y/self.yL))/self.xL
                dFdy += (np.pi*m*coeffs[n,m]*np.sin(n*np.pi*x/self.xL)*np.cos(m*np.pi*y/self.yL))/self.yL
        self.F = F
        self.dFdx = dFdx
        self.dFdy = dFdy

    def sample_patch(self,patch_size:Tuple[int,int])->Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random patch of specified size from the global fourier field.

        Args:
        patch_size(tuple of ints (N,M)): patch size

        Returns:
        F_patch(NxM float array): Fourier field values inside the patch
        ex(NxM float array): x-component of Fourier field gradient normalized to a unit-vector
        ey(NxM float array): y-component of Fourier field gradient normalized to a unit-vector

         """
        Ny, Nx = self.F.shape
        # make sure patch is more than 10% away from the boundary
        bound_dist = (0.1*np.array([Nx,Ny])).astype(int)
        # sample a random window
        idx_x = np.random.randint(bound_dist[0],Nx-bound_dist[0]-patch_size[0])
        idx_y = np.random.randint(bound_dist[1],Ny-bound_dist[1]-patch_size[1])
        # extract window from global fourier field
        F_patch = self.F[idx_x:idx_x+patch_size[0],idx_y:idx_y+patch_size[1]]
        dFpdx = self.dFdx[idx_x:idx_x+patch_size[0],idx_y:idx_y+patch_size[1]]
        dFpdy = self.dFdy[idx_x:idx_x+patch_size[0],idx_y:idx_y+patch_size[1]]
        # normalize gradients to create unit vectors
        norm_mat = np.linalg.norm(np.array([dFpdx,dFpdy]),axis=0)
        ex = dFpdx/norm_mat
        ey = dFpdy/norm_mat
        return F_patch, ex, ey 


def solve_orth_phi(sDx,sDy,ex,ey):
    """Given the finite difference matrices of the mesh and principal stress orientations
    solve for a scalar field with gradients orthogonal to the orientations

    Args:
    sDx(sparse float matrix): finite difference matrix in x-direction
    sDy(sparse float matrix): finite difference matrix in y-direction
    ex(float matrix): x-components of principal stress orientations
    ey(float matrix): y-components of principal stress orientations

    Returns:
    phi_orth(float matrix): scalar field with gradients orthogonal to the princiapl stress orientations
    """
    # setup system matrices
    A = sp.vstack([sDx,sDy])
    b = np.hstack([ex.flatten(),ey.flatten()])
    Adot = A.T.dot(A)
    bdot = A.T.dot(b)
    # sparse solver
    phi_flat = sp.linalg.spsolve(Adot,bdot)
    # reshape orthogonal field into 2d
    phi_orth = phi_flat.reshape(ex.shape)
    return phi_orth

def phi2rho(phi1,phi2,W1,W2,P1,P2,upsampling_rate):
    """Project orthogonal scalar fields phi1 and phi2 onto a finer mesh using
    lamination widths W1,W2 and periodicity P1,P2. Orthogonal scalar fields can be obtained
    using the 'solve_orth_phi' function. If a scalar is used for P1,P2 periodicity is assumed
    constant throughout the domain.

    Args:
    phi1(float matrix): orthogonal scalar field in first principal stress orientation
    phi2(float matrix): orthogonal scalar field in second principal stress orientation
    W1(float matrix): lamination width in first principal stress orientation
    W2(float matrix): lamination width in second principal stress orientation
    P1(float matrix or scalar): periodicity field in first principal stress orientation
    P2(float matrix or scalar): periodicity field in second principal stress orientation
    upsampling_rate(int): constant upsampling_rate in both x and y direction of the mesh

    Returns:
    rho(float matrix): Global binary density field
    rho1(float matrix): Binary density field in first pricipal stress orientation
    rho2(float matrix): Binary density field in second pricipal stress orientation
    rho1_tilde(float matrix): Grayscale density field in first pricipal stress orientation
    rho2_tilde(float matrix): Grayscale density field in second pricipal stress orientation
    """

    Ny,Nx = phi1.shape
    usr = upsampling_rate
    # initialize interpolation schemes
    phi1_interp = RectBivariateSpline(np.linspace(0,1,Ny),np.linspace(0,1,Nx),phi1)
    phi2_interp = RectBivariateSpline(np.linspace(0,1,Ny),np.linspace(0,1,Nx),phi2)
    W1_interp = RectBivariateSpline(np.linspace(0,1,Ny),np.linspace(0,1,Nx),W1)
    W2_interp = RectBivariateSpline(np.linspace(0,1,Ny),np.linspace(0,1,Nx),W2)
    # interpolate phi and laminations widths onto finer grid
    phi1_up = phi1_interp(np.linspace(0,1,Ny*usr),np.linspace(0,1,Nx*usr))
    phi2_up = phi2_interp(np.linspace(0,1,Ny*usr),np.linspace(0,1,Nx*usr))
    W1_up = W1_interp(np.linspace(0,1,Ny*usr),np.linspace(0,1,Nx*usr))
    W2_up = W2_interp(np.linspace(0,1,Ny*usr),np.linspace(0,1,Nx*usr))
    # project phi to grayscale using a cosine wave
    rho1_tilde = 0.5+0.5*np.cos(P1*phi1_up)
    rho2_tilde = 0.5+0.5*np.cos(P2*phi2_up)
    # determine thresholding parameters based on lamination widhts
    eta1 = 0.5+0.5*np.cos(np.pi*(1-W1_up))
    eta2 = 0.5+0.5*np.cos(np.pi*(1-W2_up))
    # threshold grayscale rho_tilde field to create binary field
    rho1 = ((rho1_tilde-eta1)>0)*1
    rho2 = ((rho2_tilde-eta2)>0)*1
    # create the global rho field by adding the binary cosine waves and taking
    # the minimum value between rho1+rho2 and 1.
    rho = np.minimum(rho1+rho2,np.ones(rho1.shape))
    return rho,rho1,rho2,rho1_tilde,rho2_tilde

def generate_dataset(savepath:str,n_samples:int,max_angle:float=25.0,min_std_angle:float=0.0):
    #window_sizes = np.array([[80,80],[60,120],[40,160]]) # window size
    window_sizes = np.array([[80,80]]) # window size
    ssr = 2 # subsampling rate of theta
    if os.path.exists(savepath) is False:
        os.makedirs(savepath)

    for i in tqdm(range(n_samples)):
        # randomly choose one of the specified window sizes
        ws = window_sizes[np.random.choice(len(window_sizes))]
        # generate a new Fourier field every n'th iteration
        if i%100==0:
            nfc = np.random.choice([6,8,10])
            print("Generating new global fourier field with",str(nfc),"Fourier coefficients")
            patch_sampler = FourierFieldSampler(xL=1,yL=1,Ngrid=800,n_coeffs=nfc)
            patch_sampler.gen_rand_field()
        # sample a new patch until angular constraint(s) is fulfilled
        max_delta_nbr_angle = 1e9
        while max_delta_nbr_angle>max_angle or std_delta_nbr_angle<min_std_angle:
            # sample a new patch from the Fourier field
            F_patch,ex,ey = patch_sampler.sample_patch(patch_size=ws)
            # block reduce unit-vectors according to subsampling rate
            ex_block_avg = measure.block_reduce(ex, block_size=(ssr,ssr), func=np.mean)
            ey_block_avg = -measure.block_reduce(ey, block_size=(ssr,ssr), func=np.mean)
            e_norm = np.linalg.norm(np.array([ex_block_avg,ey_block_avg]),axis=0)
            # normalize block-reduced unit vectors
            ex_block_avg = ex_block_avg/e_norm
            ey_block_avg = ey_block_avg/e_norm
            # determine maximum angular change between neighbouring unit-vectors
            angular_change_x = np.rad2deg(np.arccos(ex_block_avg[1:,:]*ex_block_avg[:-1,:] 
                                                    + ey_block_avg[1:,:]*ey_block_avg[:-1,:]))
            angular_change_y = np.rad2deg(np.arccos(ex_block_avg[:,1:]*ex_block_avg[:,:-1] 
                                                    + ey_block_avg[:,1:]*ey_block_avg[:,:-1]))  
            delta_nbr_angle = np.hstack([angular_change_x.flatten(),angular_change_y.flatten()])
            
            std_delta_nbr_angle = np.std(delta_nbr_angle)
            max_delta_nbr_angle = np.max(delta_nbr_angle)
        
        hh,ww = ex_block_avg.shape
        E1vecs = np.zeros((2,hh,ww))
        E1vecs[0] = ex_block_avg
        E1vecs[1] = ey_block_avg
        np.save(os.path.join(savepath,"E1_"+str(i)),E1vecs)
        
        E2vecs = np.zeros((2,hh,ww))
        E2vecs[0] = -ey_block_avg
        E2vecs[1] = ex_block_avg
        np.save(os.path.join(savepath,"E2_"+str(i)),E2vecs)

if __name__=='__main__':
    Fire(generate_dataset)

