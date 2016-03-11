"""
Based on what I saw here:
https://jameshensman.wordpress.com/2010/06/14/multiple-matrix-multiplication-in-numpy/

"""

import numpy as np
multiply = np.dot

class ArrayOps:
    @staticmethod
    def multiply_AsT_and_Bs(A,B,At_times_B):
        """
	    NB! Because we want A[i].T times B[i], 
	    we use A[:,:,:,np.newaxis] below. Had we wanted 
	    A[i].T times B[i], we would have used np.transpose(A,(0,2,1))[:,:,:,np.newaxis]
        """
        C = At_times_B 
        if A.ndim == 3 and B.ndim == 3:
            if len(A) != len(B)  and  len(A) != len(At_times_B) and len(A)!=1:
                raise ValueError(A.shape,B.shape,At_times_B.shape)
            if A.shape[1] != B.shape[1]:
                raise ValueError(A.shape,B.shape,At_times_B.shape)
            if C.shape[2] != B.shape[2]:
                raise ValueError(A.shape,B.shape,C.shape)  
            if C.shape[1] != A.shape[2]:
                raise ValueError(A.shape,B.shape,C.shape) 
            At_times_B[:] = (A[:,:,:,np.newaxis]*B[:,:,np.newaxis,:] ).sum(axis = 1)
		#[multiply(a,b,out=c) for (a,b,c) in zip(A,B,At_times_B)
        elif A.ndim == 2 and B.ndim == 3:
        #At_times_B[:] = (A[np.newaxis,:,:,np.newaxis]*B[:,:,np.newaxis,:] ).sum(axis = 1)
            At_times_B.transpose(1,0,2)[:]= multiply(A.T,B) 
		#[multiply(A.T,b,out=c) for (b,c) in zip(B,At_times_B)]
        else:
            raise NotImplementedError(A.shape,B.shape,At_times_B.shape)
	    return At_times_B
    @staticmethod	
    def multiply_As_and_Bs(A,B,A_times_B):
        """
        NB! Because we want A[i] times B[i], 
        we use np.transpose(A,(0,2,1))[:,:,:,np.newaxis] below.  
        """
        C = A_times_B
        if A.ndim == 3 and B.ndim == 3:
            if len(A) != len(B)  and  len(A) != len(At_times_B) and len(A)!=1:
                raise ValueError(A.shape,B.shape,At_times_B.shape)
            if A.shape[2] != B.shape[1]:
                raise ValueError(A.shape,B.shape,At_times_B.shape)
            if C.shape[2] != B.shape[2]:
                raise ValueError(A.shape,B.shape,C.shape)  
            if C.shape[1] != A.shape[1]:
                raise ValueError(A.shape,B.shape,C.shape) 
              
            A_times_B[:] = (np.transpose(A,(0,2,1))[:,:,:,np.newaxis]*B[:,:,np.newaxis,:] ).sum(axis = 1)
        elif A.ndim == 2 and B.ndim == 3:
		#A_times_B[:] = (A.T[np.newaxis,:,:,np.newaxis]*B[:,:,np.newaxis,:] ).sum(axis = 1)
            A_times_B.transpose(1,0,2)[:]= multiply(A,B)
		#[multiply(A.T,b,out=c) for (b,c) in zip(B,At_times_B)]
        elif A.ndim == 3 and B.ndim == 2:
#             """ Did not check this"""    
#            A_times_B.transpose(1,0,2)[:]= multiply(A,B)
            raise NotImplementedError((A.ndim,B.ndim))
        elif A.ndim == 3 and B.ndim == 1:
            """ Did not check this"""       
            A_times_B[:]= A.dot(B)    
        else:
            raise NotImplementedError((A.ndim,B.ndim))
	 
	    return A_times_B
	    
    
if __name__ == '__main__':
    import time
    N = 1000000
    dtype = np.float32
    if 0:
        A = np.random.random([3,3]).astype(dtype)
        B = np.random.random([N,3,3]).astype(dtype)
        At_times_B = np.zeros((B.shape[0],A.shape[1],B.shape[1]),dtype = B.dtype)
        At_times_B2 = np.zeros_like(At_times_B)
        tic = time.clock()
        ArrayOps.multiply_AsT_and_Bs(A,B,At_times_B)
        toc = time.clock()
        print toc-tic
        
        tic = time.clock()
        At_times_B2.transpose(1,0,2)[:]= multiply(A.T,B) 
        toc = time.clock()
        print toc-tic
        print np.allclose(At_times_B,At_times_B2)
    else:
        A = np.random.random([3,3]).astype(dtype)
        B = np.random.random([N,3,3]).astype(dtype)
        A_times_B = np.zeros((B.shape[0],A.shape[1],B.shape[1]),dtype = B.dtype)
        A_times_B2 = np.zeros_like(A_times_B)
        tic = time.clock()
        ArrayOps.multiply_As_and_Bs(A,B,A_times_B)
        toc = time.clock()
        print toc-tic
        
        tic = time.clock()
        A_times_B2.transpose(1,0,2)[:]= multiply(A,B) 
        toc = time.clock()
        print toc-tic
        print np.allclose(A_times_B,A_times_B2)
        
    
    
