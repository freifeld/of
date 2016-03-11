#!/usr/bin/env python
"""
Created on Sun Oct 26 11:28:18 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
from pycuda import autoinit
import pycuda.gpuarray as gpuarray
import pycuda.elementwise as elementwise

from pycuda.elementwise import ElementwiseKernel
from pycuda import cumath
from pycuda.compiler import SourceModule
import numpy as np

mod = SourceModule("""
__global__ void gpu_exp(const double *x, double *out, int nPts)
{
  const int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i>nPts)
      return;
  out[i] = exp(x[i]);   
}
__global__ void gpu_log(const double *x, double *out, int nPts)
{
  const int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i>nPts)
      return;  
  out[i] = log(x[i]);   
}


__global__ void gpu_normalize_2d_vectors(const double *x, double *out, int nPts)
{
  const int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i>nPts)
      return;
  // copy the values
  out[i*2+0] =  x[i*2+0];
  out[i*2+1] =  x[i*2+1];
  // Compute the magnitude
  double mag = sqrt( x[i*2+0]*x[i*2+0] + x[i*2+1]*x[i*2+1]);
  if (mag>0){
      out[i*2+0] /= mag;
      out[i*2+1] /= mag;
  }   
  
}



""")

gpu_exp = mod.get_function("gpu_exp")
gpu_log = mod.get_function("gpu_log")
gpu_normalize_2d_vectors = mod.get_function("gpu_normalize_2d_vectors")


# TODO: chose these number in some smart(er) way...




class BasicOperations:
    @staticmethod
    def exp(x,out):
        """
        x, out: CpuGpuArrays
        """
        nPts=x.shape[0]
        threadsPerBlock=512
        nBlocks = int(np.ceil(float(nPts) / float(threadsPerBlock))) 
        print 'nBlocks',nBlocks
        print 'threadsPerBlock',threadsPerBlock
        gpu_exp(x.gpu,out.gpu,np.int32(nPts), grid=(nBlocks,1,1), 
                              block=(threadsPerBlock,1,1))

    @staticmethod
    def log(x,out):
        """
        x, out: CpuGpuArrays
        """
        nPts=x.shape[0]
        threadsPerBlock=512
        nBlocks = int(np.ceil(float(nPts) / float(threadsPerBlock)))
        gpu_log(x.gpu,out.gpu,np.int32(nPts), grid=(nBlocks,1,1), 
                  block=(threadsPerBlock,1,1))        

    @staticmethod
    def normalize_2d_vectors(x,out):
        """
        x, out: CpuGpuArrays
        """
        nPts=x.shape[0]
        threadsPerBlock=512
        nBlocks = int(np.ceil(float(nPts) / float(threadsPerBlock)))
        gpu_normalize_2d_vectors(x.gpu,out.gpu,np.int32(nPts), grid=(nBlocks,1,1), 
                                 block=(threadsPerBlock,1,1))



if __name__ == "__main__":
    from of.gpu import CpuGpuArray    
    from pylab import plt    
    x = CpuGpuArray(np.linspace(-1,3,1000))
    y = CpuGpuArray.zeros_like(x)
    log_y =   CpuGpuArray.zeros_like(x)  
    
    BasicOperations.exp(x,y)
    y.gpu2cpu()
    BasicOperations.log(y,log_y)
    log_y.gpu2cpu()

    
    plt.figure(1)
    plt.clf()
    plt.plot(x.cpu,y.cpu)
    plt.plot(x.cpu,log_y.cpu)
    plt.grid('on')    
    plt.legend(['y','log(y)'])
    
    
