#!/usr/bin/env python
"""
Created on Sun May 25 09:24:56 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import numpy as np
from of.gpu.init_the_device_if_needed import init_the_device_if_needed

from pycuda.reduction import ReductionKernel
init_the_device_if_needed()

krnl_dot_product = ReductionKernel(np.float64, neutral="0",
                               reduce_expr="a+b", map_expr="x[i]*y[i]",
                                arguments="double *x, double *y")  

if __name__ == "__main__":
    import time
    from pycuda import gpuarray
    N = 100000000
    print 'generating data'
    a = np.random.standard_normal(N)
    b = np.random.standard_normal(N)
    
    print 'pushing to gpu'
    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    
    tic = time.clock()
    res_gpu = krnl_dot_product(a_gpu,b_gpu)    
    toc = time.clock()
    print 'res_gpu:',res_gpu
    print 'time',toc-tic
    
    tic = time.clock()
    res_cpu = (a*b).sum()  
    toc = time.clock()
    print 'res_cpu:',res_cpu    
    print 'time',toc-tic    