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


_kernel="""
__global__ void resampler(
double* pts_inv,
double* signal,
double* signal_wrapped,       
int nPts,
int nx,
//int ny,
int nChannels)
{
    
    int idx = threadIdx.x + blockIdx.x*blockDim.x; 
    if (idx>=nPts)
        return;                                                       
    double x = pts_inv[idx];    
    int x0 = floor(x);
    int x1 = x0 + 1;     

 
                     
    if (x0<0)
        x0=x1=0; // extend the boundary condition
        //return;
    if (x1>=nx)
        x0=x1=nx-1; // extend the boundary condition
        //return;            
    int idx_in_orig_0 = x0;     
    int idx_in_orig_1 = x1; 

    double f0,f1;       
    double xx = x-x0;    
    double new_val=0;            
    
    for (int i=0;i < nChannels; i++){          
       f0 = signal[idx_in_orig_0*nChannels + i];
       f1 = signal[idx_in_orig_1*nChannels + i];                         
    
       //new_val=f00*(1-xx)*(1-yy)+f10*xx*(1-yy)+f01*(1-xx)*yy+f11*xx*yy;                   
       new_val = f0 + (f1-f0)*xx;
       signal_wrapped[idx*nChannels + i]= new_val;                                
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
     
def resampler(pts_inv_gpu,             
          signal_gpu,
          signal_wrapped_gpu,                         
          nPts=None, 
          threadsPerBlock=1024):
        """
        Input:           
            pts_inv_gpu: a gpu_array. shape: (nPts,1). dtype: np.float64
            signal_gpu: a gpu array. dtype=np.float64 
            signal_warpped_gpu: a gpu array. dtype=np.float64        
            nPts = number of points  

        //////////////////// IMPORTANT //////////////////////
        // The values of pts_inv_gpu stand for T^{-1}(x)       //
        // and usually do not fall on integer locations.   //
        /////////////////////////////////////////////////////
            
        This function serves a similar purpose to cv2.remap, except that:
            1) it works on gpu data
            2) it works on 1D signals
        Currently, only the linear method is implemented.
        
        This function warps signal_gpu to signal_wrapped_gpu.
        Let T denote the transformation such that if x is an integer point 
        in the (time) domain of the first signal 
        (signal_gpu), 
        then (x') = T(x) is a point in the domain of second signal 
        (signal_wrapped_gpu). 
        
        The warpping is done using the *inverse* of T, not T itself. 
        
        In effect, signal_warpped_gpu(x)= signal_gpu( T^{-1}(x)).
        

        
        Remark: for locations that fall outside
                the domain, the boundary values are duplicated.  

        
        
        
        """
        
        do_checks = True
        if do_checks:
            if not isinstance(pts_inv_gpu,gpuarray.GPUArray):
                raise TypeError(type(pts_inv_gpu))                      
            if not isinstance(signal_gpu,gpuarray.GPUArray):
                raise TypeError(type(signal_gpu)) 
            if not isinstance(signal_wrapped_gpu,gpuarray.GPUArray):
                raise TypeError(type(signal_wrapped_gpu))             
                
            if pts_inv_gpu.shape[1] !=1:
                raise ValueError(pts_inv_gpu.shape)
            
            if signal_gpu.dtype != np.float64:
                raise ValueError(signal_gpu.dtype)
            if signal_wrapped_gpu.dtype != np.float64:
                raise ValueError(signal_wrapped_gpu.dtype)            
    
            if signal_gpu.shape != signal_wrapped_gpu.shape:
                raise ValueError(signal_gpu.shape , signal_wrapped_gpu.shape)        
        
        
#        try:
#            nChannels=signal_gpu.shape[2]
#        except IndexError:
#            nChannels = 1
        if len(signal_gpu.shape)!=2:
            raise ValueError(signal_gpu.shape)
        nChannels=1

        nx = signal_gpu.shape[0]
    
    
        if nPts is None:
            nPts = pts_inv_gpu.shape[0]
    
        nBlocks = int(np.ceil(float(nPts) / float(threadsPerBlock)))               
        _resampler(pts_inv_gpu,             
                  signal_gpu,
                  signal_wrapped_gpu,                         
                  np.int32(nPts),
                  np.int32(nx),
                 # np.int32(ny),
                  np.int32(nChannels),
                  grid=(nBlocks,1,1), 
                  block=(threadsPerBlock,1,1))           

 
if __name__ == '__main__':
    from of.gpu import CpuGpuArray
    
    N=1000
    t1 = CpuGpuArray(np.linspace(0,N,N+1).reshape(N+1,1))
    y = CpuGpuArray(np.sin( 2 *  t1.cpu * 2*np.pi / N))
    
    
    from pylab import plt
    
#    plt.close('all')
    plt.figure(1)
    plt.clf()
    plt.plot(t1.cpu,y.cpu,lw=20,color='b')
    
    t2 = CpuGpuArray(N*(t1.cpu/N)**4)
#    t2 = CpuGpuArray(N*(t1.cpu/N)/2) # I was probably trying some thing here...
                                     # but looks like I ended up just dividing by 2
    
    plt.plot(t2.cpu,y.cpu,'r',lw=20)
    
    
    y2=CpuGpuArray.zeros_like(y)
    
    resampler(pts_inv_gpu=t2.gpu,             
          signal_gpu=y.gpu,
          signal_wrapped_gpu=y2.gpu,                         
          nPts=None)

    y2.gpu2cpu()
    plt.plot(t2.cpu,y2.cpu,'g',lw=8)    
    plt.plot(t1.cpu,y2.cpu,'m',lw=8)
    
    plt.legend(['$y(t)$',"$y(t')=(y\circ T)(t)$",
                "$y_2(t')=(y\circ T)(t_1)$",'$y_2(t)$'])





