#!/usr/bin/env python
"""
Created on Sun Dec  7 15:09:45 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""



import numpy as np
from pycuda.compiler import SourceModule
from pycuda.driver import Context
from pycuda import gpuarray
from of.utils import ipshell

_kernel="""
__global__ void resampler(
double* pts,
double* img,
double* img_wrapped,       
int nPts,
int nx,
int ny,
int nz,
int nChannels)
{
    //int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x*blockDim.x; 
    if (idx>=nPts)
        return;                                       
    double x = pts[idx*3+0];
    double y = pts[idx*3+1];   
    double z = pts[idx*3+1];              
    
    int x0 = floor(x);
    int y0 = floor(y);   
    int z0 = floor(z);        
    int x1 = x0 + 1;        
    int y1 = y0 + 1;        
    int z1 = z0 + 1;        
    
    
    if (x0<0)
        return;
    if (x1>=nx)
        return;     
    if (y0<0)
        return;
    if (y1>=ny)
        return;      
    if (z0<0)
        return;
    if (z1>=nz)
        return; 
    
    int idx_in_orig_000;     
    int idx_in_orig_001; 
    int idx_in_orig_010; 
    int idx_in_orig_011;     
    int idx_in_orig_100;     
    int idx_in_orig_101; 
    int idx_in_orig_110; 
    int idx_in_orig_111; 
    
    double f000,f001,f010,f011,f100,f101,f110,f111;   
    double c00,c01,c10,c11; 
    double c0,c1;
    double c=0;
     
    double xx = x-x0;
    double yy = y-y0;    
    double zz = z-z0;    
     
    
    // Order is idx_in_orig_zyx
    
    idx_in_orig_000 = x0 + y0 * nx + z0 * nx * ny;        
    idx_in_orig_001 = x0 + y1 * nx + z0 * nx * ny;
    idx_in_orig_010 = x1 + y0 * nx + z0 * nx * ny;
    idx_in_orig_011 = x1 + y1 * nx + z0 * nx * ny;      

    idx_in_orig_100 = x0 + y0 * nx + z1 * nx * ny;        
    idx_in_orig_101 = x0 + y1 * nx + z1 * nx * ny;
    idx_in_orig_110 = x1 + y0 * nx + z1 * nx * ny;
    idx_in_orig_111 = x1 + y1 * nx + z1 * nx * ny;
    
    
    for (int i=0;i < nChannels; i++){          
       f000 = img[idx_in_orig_000*nChannels + i];
       f001 = img[idx_in_orig_001*nChannels + i];
       f010 = img[idx_in_orig_010*nChannels + i];
       f011 = img[idx_in_orig_011*nChannels + i];                     

       f100 = img[idx_in_orig_100*nChannels + i];
       f101 = img[idx_in_orig_101*nChannels + i];
       f110 = img[idx_in_orig_110*nChannels + i];
       f111 = img[idx_in_orig_111*nChannels + i]; 
       
       // Interpolate x
       c00 = f000*(1-xx) + f001*xx;
       c10 = f010*(1-xx) + f011*xx;
       c01 = f100*(1-xx) + f101*xx;
       c11 = f110*(1-xx) + f111*xx;
       
       // Interpolate y       
       c0 = c00*(1-yy)+c10 * yy;
       c1 = c01*(1-yy)+c11 * yy;
       
       // Interpolate z
        c = c0*(1-zz) + c1 * zz;           
       
       img_wrapped[idx*nChannels + i]= c;               
               
    }  
     
    return;        
}
"""

 

 
try:            
    Context.get_device() 
except:
    import pycuda.autoinit
mod = SourceModule(_kernel)
_resampler = mod.get_function("resampler")                
     
def resampler(pts_gpu,             
          img_gpu,
          img_wrapped_gpu,                         
          nPts=None, 
          threadsPerBlock=1024):
        """
        This function serves a similar purpose to cv2.remap,
        but works with gpu data. And in 3D.
        Currently, only the bilinear method is implemented.
        
        This function warps img_gpu to img_wrapped_gpu.
        Let T denote the transformation such that if (x,y,z) is a point 
        in the domain of the first image (img_gpu), 
        then (x',y',z') = T(x,y,z) is a point in the domain of second image 
        (img_wrapped_gpu). 
        
        The warpping is done using the *inverse* of T, not T itself. 
        
        In effect, img_warpped_gpu(x,y,z)= img_gpu( T^{-1}(x,y,z)).
        
            Note that in the line above the order is xyz, as is done
            in mathematical notation. However, of course that in terms 
            of code we use img_gpu[z,y,x] (with square brackets).
            
        Input:           
            pts_gpu: a gpu_array. shape: (nPts,3). dtype: np.float64
            img_gpu: a gpu array. dtype=np.float64 (not uint8!)
            img_warpped_gpu: a gpu array. dtype=np.float64 (not uint8!)            
            nPts = number of points (i.e., number of pixels)
        
        
        
        """
        
        do_checks = True
        if do_checks:
            if not isinstance(pts_gpu,gpuarray.GPUArray):
                raise TypeError(type(pts_gpu))                      
            if not isinstance(img_gpu,gpuarray.GPUArray):
                raise TypeError(type(img_gpu)) 
            if not isinstance(img_wrapped_gpu,gpuarray.GPUArray):
                raise TypeError(type(img_wrapped_gpu))             
                
            if pts_gpu.shape[1] !=3:
                raise ValueError(pts_gpu.shape)
            
            if img_gpu.dtype != np.float64:
                raise ValueError(img_gpu.dtype)
            if img_wrapped_gpu.dtype != np.float64:
                raise ValueError(img_wrapped_gpu.dtype)            
    
            if img_gpu.shape != img_wrapped_gpu.shape:
                raise ValueError(img_gpu.shape , img_wrapped_gpu.shape)        
        
        
#        try:
#            nChannels=img_gpu.shape[2]
#        except IndexError:
#            nChannels = 1
        nChannels = 1

        ny,nx,nz=img_gpu.shape  
        
        if nx in (1,2):
            raise ValueError(nx,"I am pretty sure this is not what you want")
    
        if nPts is None:
            nPts = pts_gpu.shape[0]
    
        nBlocks = int(np.ceil(float(nPts) / float(threadsPerBlock)))               
        _resampler(pts_gpu,             
                  img_gpu,
                  img_wrapped_gpu,                         
                  np.int32(nPts),
                  np.int32(nx),
                  np.int32(ny),
                  np.int32(nz),
                  np.int32(nChannels),
                  grid=(nBlocks,1,1), 
                  block=(threadsPerBlock,1,1))           
        
        
    








