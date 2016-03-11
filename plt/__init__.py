from pylab import plt as _plt
def axis_ij(g=None):
    if g is None:
        g = _plt.gca()
    bottom, top = g.get_ylim()  
    if top>bottom:
        g.set_ylim(top, bottom)
    else:
        pass
def axis_xy(g=None):
    if g is None:
        g = _plt.gca()
    bottom, top = g.get_ylim()  
    if top<bottom:
        g.set_ylim(top,bottom)
    else:
        pass    
    
def maximize_figure(fig=None):
    if fig is None:
        fig = _plt.gcf()
    
    mng = _plt.get_current_fig_manager()
    try:        
        mng.frame.Maximize(True)      
    except AttributeError:
        print "Failed to maximize figure."