#!/usr/bin/env python
"""
Matrix-vector multiplication using blocks of 256 threads
(and global memory). 

It is designed for computing x = A.dot(b) where:
    A is "skinny-tall"  
    b is a vector
    # of rows in A is large (e.g., numnber of pixels in an image)
    # of columns in A (or rows in b) is not too large (e.g., 100 or even 1000
    # is fine, but 10000 it will be too slow).

Created on Fri May  9 10:24:08 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import time
import numpy as np
from pycuda import  compiler, gpuarray

from of.gpu.init_the_device_if_needed import init_the_device_if_needed

class MatTimesVec(object):
    """
     When an object of this class is called (using __call__), it computes 
     A_gpu.dot(b_gpu) and stores the result in out_gpu where 
     A_gpu.shape = (self.nRowsA,self.nColsA)
     b_gpu.shape = (self.nColsA,)
     out_gpu.shape = (self.nRowsA,)      
    """
    init_the_device_if_needed()
    
    __kernel = """
    __global__ void mat_vec_krnl(const double* A,const double* b,double* x, const int nRowsA)
    {    
      //int tx = threadIdx.x;
      //const int tid = threadIdx.x;
      const int idx = threadIdx.x + blockIdx.x*blockDim.x;
    
      if(idx < nRowsA)
      {
        double Ab_col = 0;        
        const double * Acol = A+idx*N_COLS;
#pragma unroll
        for (int k = 0; k < N_COLS; ++k) {
            Ab_col += Acol[k] * b[k];
        }
        // Write to device memory;
        x[idx] = Ab_col;
      }
    }
    """
    def __init__(self,nRowsA,nColsA,my_dtype=np.float64):
        """
        Initializes a MatTimesVec object. 
        When the object is called, it computes 
        A_gpu.dot(b_gpu) and stores the result in out_gpu where 
        A_gpu.shape = (self.nRowsA,self.nColsA)
        b_gpu.shape = (self.nColsA,)
        out_gpu.shape = (self.nRowsA,)               
        """
        self.nRowsA = nRowsA
        self.nColsA = nColsA
        self._kernel = self.__kernel
#        self._kernel =(' '*4 + '#define N_ROWS {} // seems we do not use it since we pass N\n'.format(nRowsA) 
#                       +self._kernel)
        self._kernel =(' '*4 + '#define N_COLS {}\n'.format(nColsA)        
                       +self._kernel)
        
        if my_dtype == np.float64:
            pass
        elif my_dtype == np.float32:
            self._kernel = self._kernel.replace('double','float')                        
        else:
            raise NotImplementedError
            
                 
               
        
        # compile the kernel code 
        mod = compiler.SourceModule(self._kernel)                

        # get the kernel function from the compiled module
        matrixmul = mod.get_function("mat_vec_krnl")
        threadsPerBlock = 256 
        nBlocks = int(np.ceil(float(nRowsA) / float(threadsPerBlock))) 

        self._matrixmul = matrixmul                
        self._nBlocks = nBlocks
        self._threadsPerBlock = threadsPerBlock 
        self._my_dtype = my_dtype
    def __repr__(self):
        s="A wrapper around the cuda kernel below.\n\n"             
        return s + self._kernel
    def __call__(self,A_gpu,b_gpu,out_gpu,do_checks=True):
        """
        Computes A_gpu.dot(b_gpu).         
        A_gpu.shape = (self.nRowsA,self.nColsA)
        b_gpu.shape = (self.nColsA,)
        out_gpu.shape = (self.nRowsA,)
        
        Modifies out_gpu.        
        """
        
        nRowsA,nColsA = self.nRowsA,self.nColsA
         
        if do_checks:
            #types
            if not isinstance(A_gpu,gpuarray.GPUArray):
                raise TypeError(type(A_gpu))
            if not isinstance(b_gpu,gpuarray.GPUArray):
                raise TypeError(type(b_gpu))
            if not isinstance(out_gpu,gpuarray.GPUArray):
                raise TypeError(type(out_gpu)) 
            # nDims
            if len(A_gpu.shape)!=2:
                raise ValueError(A_gpu.shape)       
            if len(b_gpu.shape)!=1:
                raise ValueError(b_gpu.shape)    
            if len(out_gpu.shape)!=1:
                raise ValueError(out_gpu.shape)   
            # shapes                         
            if A_gpu.shape != (nRowsA,nColsA):
                raise ValueError(A_gpu.shape , (nRowsA,nColsA))
            if len(b_gpu) != nColsA:
                raise ValueError(len(b_gpu) , nColsA)
            if len(out_gpu) != nRowsA:
                raise ValueError(len(out_gpu) , nRowsA)  
            # dtypes    
            my_dtype=self._my_dtype
            if my_dtype != A_gpu.dtype:
                raise ValueError(my_dtype , A_gpu.dtype)
            if my_dtype != b_gpu.dtype:
                raise ValueError(my_dtype , b_gpu.dtype)
            if my_dtype != out_gpu.dtype:
                raise ValueError(my_dtype , out_gpu.dtype)                

        nBlocks = self._nBlocks
        threadsPerBlock = self._threadsPerBlock
        # Now to the actual work.
        self._matrixmul(
            # inputs
            A_gpu, b_gpu, 
            # output
            out_gpu,
            # parameter
            np.int32(nRowsA),
            # params for pycuda
            grid = (nBlocks,1,1),block = (threadsPerBlock,1,1))       

 
if __name__ == "__main__":
    
    nRowsA,nColsA = 640*48,100
    my_dtype = np.float64
        
    compute_cpu=True 
    
    # create a random matrix and a random vector
    A_cpu = np.random.randn(nRowsA,nColsA).astype(my_dtype)
    b_cpu = np.random.randn(nColsA).astype(my_dtype)
    
    out_cpu = np.empty(nRowsA)
    if compute_cpu:
        tic = time.clock()
        # compute reference on the CPU to verify GPU computation
        
        np.dot(A_cpu, b_cpu,out=out_cpu)
        toc = time.clock()
        cpu_time = toc - tic
    
    # transfer host (CPU) memory to device (GPU) memory 
    A_gpu = gpuarray.to_gpu(A_cpu) 
    b_gpu = gpuarray.to_gpu(b_cpu)
        
     
    
    # create empty gpu array for the result  
    out_gpu = gpuarray.empty(nRowsA, my_dtype)
    
    
    
    mat_times_vec = MatTimesVec(nRowsA,nColsA,my_dtype=my_dtype) 
    
    print "Computing A times b"
    print 'A.shape:',A_gpu.shape 
    print 'b.shape:',b_gpu.shape 
     
    print 'Calling the GPU code'
    tic = time.clock()
    nIterations = 1000
    for i in range(nIterations):
        if i%200==0:
            print 'iter=',i
        mat_times_vec(A_gpu,b_gpu,out_gpu,do_checks=True)
    
     
    toc = time.clock()
    print 'Done'
    gpu_time = (toc-tic)/nIterations
    
    print 'GPU time (mean over #iterations={0}):'.format(nIterations),gpu_time
    
    if compute_cpu:
        print 'CPU time:',cpu_time
        print "cpu_time / gpu_time:",cpu_time/gpu_time
    
     
     
     
    # print the results
    #print "-" * 80
    #print "Matrix A (GPU):"
    #print A_gpu.get()
    
    #print "-" * 80
    #print "Matrix B (GPU):"
    #print b_gpu.get()
    
    #print "-" * 80
    #print "Matrix C (GPU):"
    #print out_gpu.get()
     
    if compute_cpu:
        print "-" * 80
#        print "CPU-GPU difference:"
#        print out_cpu - out_gpu.get()
        
        print 
        tic =time.clock()
        out_gpu_get = out_gpu.get()
        toc =time.clock()
        print "Time for getting result back to cpu:",toc-tic
        print
        print 'np.allclose(out_cpu, out_gpu.get()) =',np.allclose(out_cpu, out_gpu_get)