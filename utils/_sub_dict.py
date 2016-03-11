#! /usr/bin/env python
"""
Create a subdict. 

I don't remember where I took this from... Sorry. 

Created on Tue Feb 11 09:53:27 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
def sub_dict(somedict,somekeys,default = None):
	return dict([(k,somedict.get(k,default)) for k in somekeys])
 

        
        
        
        
