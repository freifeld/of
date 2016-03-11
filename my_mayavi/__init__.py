#!/usr/bin/env python
"""
Created on Sat Jan 26 13:28:52 2013

Author: Oren Freifeld
Email: freifeld@dam.brown.edu
"""

try:    
    from mayavi.mlab import triangular_mesh as mayavi_mlab_triangular_mesh
    from mayavi.mlab import close as mayavi_mlab_close
    from mayavi.mlab import figure as mayavi_mlab_figure
    from mayavi.mlab import gcf as mayavi_mlab_gcf
    from mayavi.mlab import clf as mayavi_mlab_clf
    from mayavi.mlab import view as mayavi_mlab_view
    from mayavi.mlab import savefig as mayavi_mlab_savefig
    from mayavi.mlab import orientation_axes as mayavi_orientation_axes
    from mayavi.mlab import plot3d as mayavi_mlab_plot3d
    from mayavi.mlab import screenshot as mayavi_mlab_screenshot
    
    mayavi_mlab_close_all = lambda : mayavi_mlab_close(all=True)
    
    
     
        
    def mayavi_mlab_figure_bgblack(*arg,**kw):
        black = 0,0,0
        kw['bgcolor']=black    
        return mayavi_mlab_figure(*arg,**kw)
    
    def mayavi_mlab_figure_bgwhite(*arg,**kw):
        white = 1,1,1
        kw['bgcolor']=white    
        return mayavi_mlab_figure(*arg,**kw)    
    
    
    def mayavi_mlab_set_parallel_projection(tf):
        tf = bool(tf) # An int (e.g., tf=1), won't cut it. mayavi wants a bool.
        f = mayavi_mlab_gcf()
        f.scene.parallel_projection = tf
    
except ImportError:
    pass
    
