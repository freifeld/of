#!/usr/bin/env python
"""
Command line usage: Exactly one argument, <filename>, is required.

Created on Tue Feb 11 09:53:27 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import os
import glob
import shutil
import sys
from datetime import datetime
import time

def logger(fullfilename,fullfilename_prev_ver):    
    log_filename = fullfilename + '.log_versions'     
    now = datetime.now()
    t = '/'.join([str(now.year)] +
                    ['{0:02}'.format(item) for item in
                     (now.month,now.day)])
    t += ' at '
    t += ':'.join([str(now.year)] +
                    ['{0:02}'.format(item) for item in
                     (now.hour,now.minute)])    
    with open( log_filename,'a') as O:
        line =  os.path.split(fullfilename_prev_ver)[1] + ' copy was created on ' + t        
        O.write(line + '\n')
        
def save_previous_version(fullfilename,verbose = True):
    """    
    If <fullfilename> exists, the function will create a copy of it
    named <fullfilename.v_num> according to the following rule:
    if no <fullfilename.v_num> exists, then num will be 001.
    Otherwise, num will be incremented accordingly. 
    """
    if os.path.isfile(fullfilename):
            # get list of all files ending with .v*
            L = glob.glob( fullfilename + '.v*')
            if verbose and len(L):
                print L

            # incrmenting the version number.
            fullfilename_prev_ver = fullfilename + '.v_{0:03}'.format(len(L)+1)
            if verbose:
                print fullfilename_prev_ver
                
            #                   src            dst 
            shutil.copyfile(fullfilename,fullfilename_prev_ver)
            #
            logger(fullfilename,fullfilename_prev_ver)
    elif verbose:
        print 'Could not find file {0}.'.format(fullfilename)

def main(argv = None):
    if argv is None:
        argv = sys.argv[1:]  
    def command_line_usage():
        print '\n'*2 + __doc__    
     
    if len(argv) != 1:
        command_line_usage()
        return
    
    filename = argv[0]     
    save_previous_version(filename , verbose = True)
    
     
if __name__ == "__main__":
    main()    
