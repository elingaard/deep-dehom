import os
from typing import Tuple

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter
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

def sample_rand_uniform_field(size:Tuple[int,int])->Tuple[np.ndarray,np.ndarray]:
    """Sample a uniform field of the specified size with a constant value between [-pi;pi]"""
    theta = np.random.uniform(-np.pi,np.pi)
    ex = np.ones(size)*np.cos(theta)
    ey = np.ones(size)*np.sin(theta)
    return ex,ey

def sample_connected_component_patch(patch_size:Tuple[int,int],dens_range:Tuple[float,float])->np.ndarray:
    """Sample a random region from a connected component analysis on a random field
    and smooth it using a Gaussian"""
    valid_labels = np.array([])
    while valid_labels.size==0:
        Z = np.random.random(patch_size)
        Z = gaussian_filter(Z,sigma=5)
        Z = normalize_data(Z)
        Zthres = Z>0.5
        label_img, nr_labels = measure.label(Zthres,background=0,return_num=True)
        # count number of pixels associated with each label
        cnt_labels = np.array([np.sum(label_img==i) for i in range(nr_labels+1)])
        # get density of each region (except background)
        density_ratio = cnt_labels[1:]/(patch_size[0]*patch_size[1])
        # pick a random label between labels with a density inside the specified range
        dens_idx = np.logical_and(density_ratio>dens_range[0],density_ratio<dens_range[1])
        valid_labels = np.arange(1,nr_labels+1)[dens_idx]
    label_idx = np.random.choice(valid_labels)
    # get background mask
    background_mask = np.logical_and(label_img!=0,label_img!=label_idx)
    # set all other labels to zero
    dens_field = label_img.copy()
    dens_field[background_mask] = 0
    # convert to zero-one
    dens_field[dens_field>0] = 1
    dens_field = dens_field.astype(np.float64)
    # smooth the discrete field
    dens_field = gaussian_filter(dens_field,sigma=10)
    return dens_field

def gen_finite_diff_mat(nx,ny,deriv_dir):
    """Generate finite difference matrix in the specified direction. First-order central difference is used for interior points,
    while first-order back/forward difference is used at boundaries.

    Args:
    nx(int): number of nodes in x-direction
    ny(int): number of nodes in y-direction
    deriv(str): string indicating finite difference direction, "x" or "y"

    Returns:
    sD(float matrix): sparse finite difference matrix
    """

    # create index matrix and pad it with -1 used to indicate the edge
    nnodes = nx*ny
    idx_mat = np.arange(nnodes).reshape(ny,nx)
    idx_mat = np.pad(idx_mat, (1, 1),'constant',constant_values=-1)
    # determine derivative direction
    if deriv_dir=="y":
        x_step = 0
        y_step = 1
    elif deriv_dir=="x":
        x_step = 1
        y_step = 0
    # initialize lists used to store sparse matrix information
    row_idx = []
    col_idx = []
    mat_data = []
    cnt = 0
    for j in range(1,ny+1):
        for i in range(1,nx+1):
            node_c = idx_mat[j,i] # center node
            node_b = idx_mat[j-y_step,i-x_step] # backward node
            node_f = idx_mat[j+y_step,i+x_step] # forward node
            # forward difference
            if node_b==-1:
                row_idx+=[cnt]*2
                col_idx+=[node_c,node_f]
                mat_data+=[-1,1]
            # backward difference
            elif node_f==-1:
                row_idx+=[cnt]*2
                col_idx+=[node_b,node_c]
                mat_data+=[-1,1]
            # central difference
            else:
                row_idx+=[cnt]*3
                col_idx+=[node_b,node_c,node_f]
                mat_data+=[-1,0,1]
            cnt+=1
    # sparse matrix creation
    mat_data = np.array(mat_data)
    row_idx = np.array(row_idx).astype(int)
    col_idx = np.array(col_idx).astype(int)
    sD = sp.csc_matrix((mat_data,(row_idx,col_idx)))
    return sD

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

def evec2cell(evec,w1,w2,epsilon,Ngrid=100):
    """
    Project a single unit vectors and corresponding densities in each direction
    onto a unit cell using a cosine wave.

    Args:
    evec(2x1 float vector): unit vector
    w1(float): density in first principal orientation
    w2(float): density in second principal orientation
    epsilon(float): unit cell periodicity
    Ngrid(int): resolution of unit cell

    Returns:
    rho(float matrix): Unit cell binary density field
    rho1(float matrix): Binary density field in first pricipal stress orientation
    rho2(float matrix): Binary density field in second pricipal stress orientation
    rho1_tilde(float matrix): Grayscale density field in first pricipal stress orientation
    rho2_tilde(float matrix): Grayscale density field in second pricipal stress orientation
    """

    # create the two orthogonal basis vectors
    ex,ey = evec
    e1 = np.array([-ey,ex])
    e2 = np.array([ex,ey])
    # set periodicity
    P = 2*np.pi/epsilon
    # generate unit-cell mesh
    y,x = np.meshgrid(np.linspace(0,1,Ngrid),np.linspace(0,1,Ngrid))
    XY = np.array([x,y])
    # project unit-vectors using cosine wave
    rho1_tilde = 0.5+0.5*np.cos(P*np.matmul(XY.T,e1))
    rho2_tilde = 0.5+0.5*np.cos(P*np.matmul(XY.T,e2))
    # determine thresholds based on density
    eta1 = 0.5+0.5*np.cos(np.pi*(1-w1))
    eta2 = 0.5+0.5*np.cos(np.pi*(1-w2))
    # create binary rho_tilde fields based on density thresholds
    rho1 = ((rho1_tilde-eta1)>0)*1
    rho2 = ((rho2_tilde-eta2)>0)*1
    # create binary global density field based on rho1 and rho2
    rho = np.minimum(rho1+rho2,np.ones(rho1.shape))
    return rho,rho1,rho2,rho1_tilde,rho2_tilde

def unitcell2globalwave(Evecs,W1,W2,epsilon,Ngrid):
    """Naive mapping of each unit vector in the global vector field [Ex,Ey] onto
    on a global density field using the 'evec2cell' function.

    Args:
    Evecs(list): list containing the global x and y components of the principal stress directions
    W1(float matrix): lamination width in first principal stress orientation
    W2(float matrix): lamination width in second principal stress orientation
    epsilon(float): unit cell periodicity
    Ngrid(int): resolution of each unit cell in the global mesh

    Returns:
    rho_canvas(float matrix): Global binary density field
    """
    # unpack global vector field
    Ex,Ey = Evecs
    nely,nelx = Ex.shape
    # initialize a canvas to insert unit cells onto
    rho_canvas = np.zeros((nely*Ngrid,nelx*Ngrid))
    cnt = 0
    # loop over each unit vector and insert in canvas
    for j in range(nely):
        for i in range(nelx):
            ex = Ex[j,i]; ey = Ey[j,i]; w1 = W1[j,i]; w2=W2[j,i];
            rho_unit,_,_,_,_,_ = evec2cell([ex,ey],w1,w2,epsilon,Ngrid)
            rho_canvas[j*Ngrid:(j+1)*Ngrid,i*Ngrid:(i+1)*Ngrid] = rho_unit
    return rho_unit

def normalize_data(x:np.ndarray)->np.ndarray:
    return (x-x.min())/(x.max()-x.min())

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

