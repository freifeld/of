#!/usr/bin/env python
"""
Created on Tue Sep 25 15:06:05 2012

Author: Oren Freifeld
Email: freifeld@dam.brown.edu
"""
import pickle
import os 
from _generic_exceptions import FileAlreadyExistsError, DirDoesNotExistError
from _FilesDirs import FilesDirs

  

class Pkl:
    @staticmethod
    def load(fullpath_filename,verbose = False):
        """
        TODO: doc
        """        
        fullpath_filename = os.path.expanduser(fullpath_filename)
        FilesDirs.verify_correct_file_ext(fullpath_filename,'pkl')        
        FilesDirs.raise_if_file_does_not_exist(fullpath_filename)
        if verbose:
            print 'Loading ' + fullpath_filename
        try:
            with open(fullpath_filename,'r') as f:
#                ret = pickle.load(f)
                return pickle.load(f)
        except:
#            print 'Failed!'
#            ipshell()
            raise
#        return ret
    @staticmethod
    def dump(fullpath_filename, x, verbose=True, override=False, create_dir_if_needed=False):
        """
        TODO: doc
        """
        FilesDirs.verify_correct_file_ext(fullpath_filename,'pkl') 
        try:
            FilesDirs.raise_if_file_already_exists(fullpath_filename)
        except FileAlreadyExistsError:
            if override:
                print 'Overriding existing file: {0}.'.format(fullpath_filename) 
            else:
                raise
        if verbose:
            print 'Saving ' + fullpath_filename
            
        try:
            dirname = os.path.dirname(fullpath_filename)                            
            FilesDirs.raise_if_dir_does_not_exist(dirname)
        except DirDoesNotExistError:
            if create_dir_if_needed:
                os.makedirs(dirname)
            else:
                raise
       
        with open(fullpath_filename,'w') as f:
            pickle.dump(x,f)
            
            
    @staticmethod
    def load_numpy_array_inplace(fullpath_filename,dst,verbose = True):
        """
        TODO: doc
        I almost never use it.
        """        
        FilesDirs.verify_correct_file_ext(fullpath_filename,'pkl') 
        FilesDirs.raise_if_file_does_not_exist(fullpath_filename)
        if verbose:
            print 'Loading ' + fullpath_filename
        
        with open(fullpath_filename,'r') as f:
            dst[:] = pickle.load(f)
         

if __name__ == "__main__":    
    s = 'abc' # some string
    x = 17 # some number
    

    my_dict = {'s':s,x:'x'} # a dictinoary. 

    # If you want want to save numpy arrays, 
    # you can simply add them to the dict. 
    # E.g.:
    #  arr = np.zeros(3)
    # my_dict = {'s':s,x:'x','arr':arr}

    
    Pkl.dump('./debugging_Pkl_dump_saving_a_dict.pkl',my_dict,override=True)
    
    # You can save other objects, you don't have to have a dict.
    # But not everything can be saved by pickle (which is called internally)
    
    Pkl.dump('./debugging_Pkl_dump_saving_variable.pkl',s,override=True)
    
    
