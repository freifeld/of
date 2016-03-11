#! /usr/bin/env python
"""
Get screen info. 
There must be a better way to do it..


Created on Tue Feb 11 09:53:27 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import os
 
#import Tkinter 
# the next lines are expensive. So call them only once here,
# and not in get_screen_width
##root = Tkinter.Tk()
##screen_width = root.winfo_screenwidth()
##screen_height = root.winfo_screenheight()
##root.destroy()
class _Screen(object):
    def __init__(self):
        self.width,self.height = self._get_screen_res()
    def __repr__(self):
        s = 'screen:\n'
        for attr in sorted(self.__dict__.keys()):
            s += '\t{0:20} =  {1}\n'.format(attr,getattr(self,attr))
        s = s[:-1]# get rid of the last \n. 
        return s 
    @staticmethod
    def _get_screen_res():
	"""
	Get screen resolution.
	"""
        if not os.name == 'posix':
            return [1000,1000] # fake numbers
            raise NotImplementedError('If you are not using linux, please write your own version for this function.')
    
        if 'SGE_O_SHELL' in os.environ:
            # we're on Brown CS's grid computer, so fake some numbers as placeholders
            return [1000,1000]
        x = os.popen("xdpyinfo  | grep 'dimensions:'").read()
        try:
            if len(x) == 0:
                raise ValueError
        except ValueError: 
            return [1000,1000]
        x = x.split(' pixels')[0]
        x = x.split(' ')[-1]
        try:
            return map(int,x.split('x'))
        except ValueError:
            print 'x:'
            print x
            raise
 
screen = _Screen()
if __name__ == '__main__':    
    print screen
