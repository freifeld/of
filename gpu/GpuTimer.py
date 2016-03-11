#!/usr/bin/env python
"""
Created on Wed Aug 19 11:56:25 2015

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

from pycuda import driver as drv

class GpuTimer(object):
    def __init__(self):
        self.tic_was_called = False
        self.toc_was_called = False
    def tic(self):
        self.tic_was_called = True
        self.start = drv.Event()
        self.end = drv.Event()
        self.start.record() # start timing
    def toc(self):
        self.end.record() # end timing
        self.toc_was_called = True
        # calculate the run length
        self.end.synchronize()
        self.secs = self.start.time_till(self.end)*1e-3 # [msec]-->[sec]
    def __repr__(self):
        if self.tic_was_called and self.toc_was_called:
            return 'GpuTimer: secs = {0}'.format(self.secs)
        else:
            return 'Unused GpuTimer'
        
        
        
        