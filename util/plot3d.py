import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np

import warnings
warnings.filterwarnings("ignore")

"""
Library uses seaborn style
label size changed to 16 - to change modify mpl.rcParams['xtic.labelsize']
legend fontsize changed to 16 - to change modify mpl.rcParams['xtic.labelsize']
"""
#general style
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16
sns.set_style("white")
sns.set_style("ticks")

def format_data(XY,Z,AP=0):
    n = int(XY.shape[0]**.5)
    X1 = XY[:,0].reshape((n,n))
    Y1 = XY[:,1].reshape((n,n))
    Z1 = Z[:,AP].reshape((n,n))
    return X1,Y1,Z1

def common_kwargs(kwargs):
    AP = kwargs.pop('AP',0)
    fig = kwargs.pop('fig',None)
    ax  = kwargs.pop('ax',None)

    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')

    return AP, fig, ax

def wireframe(XY,Z,**kwargs):
    AP, fig, ax = common_kwargs(kwargs)    
    X1, Y1, Z1 = format_data(XY,Z,AP=AP)
    ax.plot_wireframe(X1, Y1, Z1,**kwargs)
    return fig, ax

def surface(XY,Z,**kwargs):
    AP, fig, ax = common_kwargs(kwargs)    
    X1, Y1, Z1 = format_data(XY,Z,AP=AP)

    rstride = kwargs.pop('rstride',1)
    cstride = kwargs.pop('cstride',1)

    ax.plot_surface(X1, Y1, Z1, rstride=rstride, cstride=cstride, **kwargs)
    return fig, ax

def scatter(XY,Z,**kwargs):
     
    AP      = kwargs.pop('AP',0)
    label   = kwargs.pop('label',None)
    color   = kwargs.pop('c','b')
    ax      = kwargs.pop('ax',None)
    fig     = kwargs.pop('fig',None)
    legend_pos = kwargs.pop('legend',None)
    # Verify if Z is a tuple or not. If tuple check if # labels is correct
    if type(Z) is tuple:
        n = len(Z)    

        if label is not None:
            assert len(label) == len(Z)
            labels = label 
        else:
            labels = [None]*n    #generate a list of None

        if color is not 'b':
            assert len(color) == len(Z)
        else:
            color = ['b']*n         #generate a list of 'b'

    else:
        n = 0

    #if no fig, ax handler has been passed, create new ones
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')
    
    #Only one Z array
    if n==0:
        ax.scatter(XY[:,0], XY[:,1], Z[:,AP:AP+1],label=label,**kwargs)    
    #Multiple Z arrays
    else:        
        for z,l,c in zip(Z,labels,color):
            ax.scatter(XY[:,0], XY[:,1], z[:,AP:AP+1],label=l,c=c,**kwargs)

    if label is not None:
        if legend_pos is not None:
            plt.legend(legend_pos)
        else:
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

    ax.set_xlabel('x [m]',fontsize=26,fontweight='bold')
    ax.set_ylabel('y [m]',fontsize=26,fontweight='bold') 
    ax.set_zlabel('RSS [dB]',fontsize=26,fontweight='bold') 

#   ax.scatter(data['X'][:,0], data['X'][:,1],Y_pl[:,AP],c='r',label='Pathloss')
    plt.show()
