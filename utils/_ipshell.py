#!/usr/bin/env python
"""
Get ipython embedded shell. 
Useful for debugging.
You can place ipshell(some_msg_string) in your code to drop into ipython.

Every time there is a new version of ipython I find that I need to change this script... 

Created on Sat Jun 29 16:55:38 2013

Author: Oren Freifeld
Email: freifeld@dam.brown.edu
"""


 
try:
    from IPython import __version__
except:
    raise

if __version__ >= '1.0.0':
    from IPython.terminal.embed import InteractiveShellEmbed                                        
    ipshell = InteractiveShellEmbed( banner1 = 'Dropping into IPython',
                                     exit_msg = 'Leaving IPython, back to program.')
                                        
#else:
#    from IPython.frontend.terminal.embed import InteractiveShellEmbed            
#    ipshell = InteractiveShellEmbed( banner1 = 'Dropping into IPython',
#                exit_msg = 'Leaving IPython, back to program.')     

#print ipshell

if 0:
    try:
        try: # for ipython version < 0.13
            from IPython.Shell import IPShellEmbed
            ipshell = IPShellEmbed()  
        except ImportError:  # for 0.13 <= ipython version < 1.0.0
            from IPython import __version__
            if __version__ >= '1.0.0':
                from IPython.terminal.embed import InteractiveShellEmbed
                ipshell = InteractiveShellEmbed( banner1 = 'Dropping into IPython',
                                exit_msg = 'Leaving IPython, back to program.')    
            else:
                from IPython.frontend.terminal.embed import InteractiveShellEmbed            
                ipshell = InteractiveShellEmbed( banner1 = 'Dropping into IPython',
                            exit_msg = 'Leaving IPython, back to program.')                    
#            try:     
#                
#                from IPython.frontend.terminal.embed import InteractiveShellEmbed            
#                ipshell = InteractiveShellEmbed( banner1 = 'Dropping into IPython',
#                            exit_msg = 'Leaving IPython, back to program.')                          
#            except TypeError: # for 1.0.0 <= ipython version 
#                from IPython.terminal.embed import InteractiveShellEmbed
#                ipshell = InteractiveShellEmbed( banner1 = 'Dropping into IPython',
#                            exit_msg = 'Leaving IPython, back to program.')
        
    except:
	raise
        print """
        Failed importing ipython embedded shell. Pitty. 
        This is useful, but not a must. 
        """        
        ipshell=None   

    # Now do the actual call
#    ipshell(*args,**kw)
    
