#!/usr/bin/env python
"""
Author: Oren Freifeld
Email: freifeld@dam.brown.edu
"""

try:    
    import numpy as np
    if np.version.version < '1.6.1' and not np.version.version.startswith('1.10'):       
        raise ValueError("Numpy is too old:",np.version.version)
    import scipy
    if scipy.version.version < '0.10':
	raise ValueError("Scipy is too old:",scipy.version.version)         
        
except ValueError:  
	#pass
	raise

def check_matplotlib_can_save_jpgs():
    """Check matplotlib version"""
    import matplotlib
    min_version_required = '1.1'  #  0.99 was not good enough; maybe 1.0 suffices?
    if matplotlib.__version__ <= min_version_required:   
        msg = """
         matplotlib.__version__= {0} <= {1}
         Thus, pylab.imsave does not support jpg.
         If you are stranded without ability to upgrade,
         change the extension from jpg to png.             
                 """.format(matplotlib.__version__,min_version_required)
        raise ValueError(msg)
    
    
  
    

 
