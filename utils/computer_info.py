#!/usr/bin/env python
"""
Easy access to some computer info.

Created on Thu Feb 13 08:59:21 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

try:
    from of_internal.utils import computer    
except ImportError:
    from platform import architecture as _architecture
    from getpass import getuser as _getuser
    from socket import gethostname as _gethostname
    
    class _Computer(object):
        def __init__(self, do_sanity_checks = True):
            self.hostname = _gethostname()
            self.user = _getuser()                         
            
            self.is64bit = _architecture()[0] == '64bit'
            self.is32bit = _architecture()[0] == '32bit'
            if do_sanity_checks:
                # sanity check
                if self.is64bit + self.is32bit != 1:
                    raise ValueError(_architecture()[0])
               
            self.has_good_gpu_card = True # Let's be optimistic.
                                          # Change it to false if
                                          # that's not the case.
                                    
                
        def __repr__(self):
            s = 'Computer info:\n'
            first = ('hostname','user')
            second = sorted(list(set(self.__dict__.keys()).difference(first)))
            for L in (first,second):
                for attr in L:
                    if attr.startswith('_'):
                        continue
                    s += '\t{0:20} =  {1:10}\n'.format(attr,getattr(self,attr))
            s = s[:-1]# get rid of the last \n. 
            return s
         
         
    
     
    computer = _Computer(do_sanity_checks=False)
    del _Computer

if __name__ == '__main__':       
    print computer
