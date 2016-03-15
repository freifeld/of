#!/usr/bin/env python
"""
Created on Wed Jul  9 19:27:48 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import numpy as np
from pycuda.compiler import SourceModule
from pycuda.driver import Context
from pycuda import gpuarray
from of.utils import ipshell

_kernel="""
__global__ void resampler_64_64_64(
double* pts,
double* img,
double* img_wrapped,       
int nPts,
int nx,
int ny,
int nChannels)
{
    //int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x*blockDim.x; 
    if (idx>=nPts)
        return;                                       
    double x = pts[idx*2+0];
    double y = pts[idx*2+1];              
    
    int x0 = floor(x);
    int y0 = floor(y);        
    int x1 = x0 + 1;        
    int y1 = y0 + 1;        
    
    /////  BUG FIX. April 7, 2015. 
    ///// If the point happens to an integer, we don't need to interpolate.
    if (x0==int(x))
        x1=x0;
    if (y0==int(y))
        y1=y0;
    ///// END BUG FIX
    
    if (x0<0)
        return;
    if (y0<0)
        return;
        
    if (x1>=nx)
        return;   
    if (y1>=ny)
        return;  
      
    
    int idx_in_orig_00;     
    int idx_in_orig_01; 
    int idx_in_orig_10; 
    int idx_in_orig_11;     
    
    double f00,f01,f10,f11;        
    double xx = x-x0;
    double yy = y-y0;    
    double new_val=0;    
    
    idx_in_orig_00 = x0 + y0 * nx;        
    idx_in_orig_01 = x0 + y1 * nx;
    idx_in_orig_10 = x1 + y0 * nx;
    idx_in_orig_11 = x1 + y1 * nx;      
    
    for (int i=0;i < nChannels; i++){          
       f00 = img[idx_in_orig_00*nChannels + i];
       f01 = img[idx_in_orig_01*nChannels + i];
       f10 = img[idx_in_orig_10*nChannels + i];
       f11 = img[idx_in_orig_11*nChannels + i];                     
    
       new_val=f00*(1-xx)*(1-yy)+f10*xx*(1-yy)+f01*(1-xx)*yy+f11*xx*yy;                   
       img_wrapped[idx*nChannels + i]= new_val;                                
    }         
    return;        
}
__global__ void resampler_64_32_32(
double* pts,
float* img,
float* img_wrapped,       
int nPts,
int nx,
int ny,
int nChannels)
{
       
    //int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x*blockDim.x; 
    if (idx>=nPts)
        return;                                       
    double x = pts[idx*2+0];
    double y = pts[idx*2+1];              
    
    int x0 = floor(x);
    int y0 = floor(y);        
    int x1 = x0 + 1;        
    int y1 = y0 + 1;        
    
    /////  BUG FIX. April 7, 2015. 
    ///// If the point happens to an integer, we don't need to interpolate.
    if (x0==int(x))
        x1=x0;
    if (y0==int(y))
        y1=y0;
    ///// END BUG FIX
    
    if (x0<0)
        return;
    if (y0<0)
        return;
        
    if (x1>=nx)
        return;   
    if (y1>=ny)
        return;  
      
    
    int idx_in_orig_00;     
    int idx_in_orig_01; 
    int idx_in_orig_10; 
    int idx_in_orig_11;     
    
    float f00,f01,f10,f11;        
    double xx = x-x0;
    double yy = y-y0;    
    float new_val=0;    
    
    idx_in_orig_00 = x0 + y0 * nx;        
    idx_in_orig_01 = x0 + y1 * nx;
    idx_in_orig_10 = x1 + y0 * nx;
    idx_in_orig_11 = x1 + y1 * nx;      
       
    for (int i=0;i < nChannels; i++){          
       f00 = img[idx_in_orig_00*nChannels + i];
       f01 = img[idx_in_orig_01*nChannels + i];
       f10 = img[idx_in_orig_10*nChannels + i];
       f11 = img[idx_in_orig_11*nChannels + i];                     
    
       new_val=f00*(1-xx)*(1-yy)+f10*xx*(1-yy)+f01*(1-xx)*yy+f11*xx*yy;                   
       img_wrapped[idx*nChannels + i]= new_val;                                
    }         
    return;        
}
"""

 

 
try:            
    Context.get_device() 
except:
    import pycuda.autoinit
mod = SourceModule(_kernel)
_resampler_64_64_64 = mod.get_function("resampler_64_64_64")                
_resampler_64_32_32 = mod.get_function("resampler_64_32_32")                
     
     
def resampler(pts_gpu,             
          img_gpu,
          img_wrapped_gpu,                         
          nPts=None, 
          threadsPerBlock=1024):
        """
        This function serves a similar purpose to cv2.remap,
        but works with gpu data. 
        Currently, only the bilinear method is implemented.
        
        This function warps img_gpu to img_wrapped_gpu.
        Let T denote the transformation such that if (x,y) is a point 
        in the domain of the first image (img_gpu), 
        then (x',y') = T(x,y) is a point in the domain of second image 
        (img_wrapped_gpu). 
        
        The warpping is done using the *inverse* of T, not T itself. 
        
        In effect, img_warpped_gpu(x,y)= img_gpu( T^{-1}(x,y)).
        
            Note that in the line above we wrote x first, then y, as is done
            in mathematical notation. However, of course that in terms 
            of code we use img_gpu[y,x] (with saqure brackets).
            
        Input:           
            pts_gpu: a gpu_array. shape: (nPts,2). dtype: np.float64
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
                
            if pts_gpu.shape[1] !=2:
                raise ValueError(pts_gpu.shape)
            if pts_gpu.dtype != np.float64:
                    raise ValueError(img_gpu.dtype,'But I expected np.float64')
            
            if img_gpu.dtype != img_wrapped_gpu.dtype:
                raise TypeError(img_gpu.dtype ,img_wrapped_gpu.dtype)
            if img_gpu.dtype not in (np.float64,np.float32):
                    raise ValueError(img_gpu.dtype,'But I expected np.float64 or np.float32')
            
            if img_gpu.dtype == np.float64:
                _resampler = _resampler_64_64_64
            elif img_gpu.dtype == np.float32:
                _resampler = _resampler_64_32_32
                

            if img_gpu.shape != img_wrapped_gpu.shape:
                raise ValueError(img_gpu.shape , img_wrapped_gpu.shape)        
        
        
        try:
            nChannels=img_gpu.shape[2]
        except IndexError:
            nChannels = 1

        ny,nx=img_gpu.shape[:2]  
        
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
                  np.int32(nChannels),
                  grid=(nBlocks,1,1), 
                  block=(threadsPerBlock,1,1))           
        
        
    








