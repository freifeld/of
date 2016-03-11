from _often_used_builtin_modules import *
from _generic_exceptions import *
from _FilesDirs import FilesDirs
from _Pkl import Pkl
from _save_previous_version import save_previous_version
from _screen import screen
from _Bunch import Bunch
from _ipshell import ipshell



 

    

 
def print_me(x):
    print(x)
    
def print_iterable(x):
    """
    Will print items in differnet lines.
    """
    map(print_me,x)     

def inside_spyder():
    return any(['SPYDER' in name for name in os.environ])




# TODO: make this cross platform.

_dirname_of_this_file = os.path.dirname(inspect.getfile(inspect.currentframe()))
HOME = os.path.expanduser('~')
DESKTOP = os.path.join(HOME,'Desktop')



import datetime 
def get_time_stamp():
    now = datetime.datetime.now()
    t = '_'.join([str(now.year)] +
                    ['{0:02}'.format(item) for item in
                     (now.month,now.day)])
    t += '_at_'
    t += '_'.join( ['{0:02}'.format(item) for item in
                     (now.hour,now.minute,now.second)])    
    return t



# I am not sure it is the best idea out there since both
# of_internal.utils and of.utils
# import from each other - and the order matters. 
# But it seems that if the lines below
# are placed at the end of this file, then it works.
#try:
#    from of_internal.utils.computer_info import *
#    print "WORKED"
#except ImportError:    
#    from computer_info import computer
    

from of_internal.utils.computer_info import computer   
