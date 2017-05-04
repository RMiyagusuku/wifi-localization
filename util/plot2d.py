import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import Tango
from mpl_toolkits.mplot3d import Axes3D


def get_from_kwargs(expected_kwargs,default_values,dict_kwargs):
  
  l = list()
  for k,v in zip(expected_kwargs,default_values):
    if k in dict_kwargs:
      l.append(dict_kwargs[k])
    else:
      l.append(v)
  return l


def GP(x,y,yvar,**kwargs):

  nsd = kwargs.get('nsd',2) 
  #color palette
  light1 =  Tango.colors('lightBlue')
  light2 =  Tango.colors('lightPurple')
  dark1  =  Tango.colors('darkBlue')
  dark2  =  Tango.colors('darkPurple')
 

  #args
  expected_kwargs = ('linewidth','axis_lim','figsize','ref','data','fontsize','capthick')
  default_values  = (3,(0,10,-2,2),(24,9),None,None,22,6)
  linewidth,axis_lim,figsize,ref,data,fontsize,capthick = get_from_kwargs(expected_kwargs,default_values,kwargs)
  
  #fig    
  fig = plt.figure(figsize=figsize) 
  ax  = fig.add_subplot(1,1,1)
  
  y_upper = y + nsd*(yvar**0.5) # +2 SD
  y_lower = y - nsd*(yvar**0.5) # -2 SD

  if ref is not None:
    y_ref = ref[0]
    yvar_ref = ref[1]     
    y_ref_upper = y_ref + nsd*(yvar_ref**0.5)
    y_ref_lower = y_ref - nsd*(yvar_ref**0.5)
  
    refz,  = plt.plot(x,y_ref,'--k',linewidth=linewidth,label='Ref. function')
    refz2, = plt.plot(x,y_ref_upper,':k',linewidth=linewidth,label='Ref. confidence interval')
    plt.plot(x,y_ref_lower,':k',linewidth=linewidth)

  expz,  = plt.plot(x,y,color=dark1,linewidth=linewidth,label='Expected value')
  plt.fill_between(x.flatten(),y_upper.flatten(),y_lower.flatten(),facecolor=light1,alpha=0.3)
  expz2, = plt.plot([], [], light1,alpha=0.3,linewidth=10,label='Confidence interval')

  if data is not None:
    X = data[0].copy().flatten()
    Y = data[1].copy().flatten()
    if len(data) > 2:
      Yvar = data[2].copy().flatten()
    else:
      Yvar = 0
    train = plt.errorbar(X, Y, yerr=(nsd*Yvar**0.5), linewidth=0, fmt='-o',ecolor=light2,capthick=capthick,elinewidth=linewidth,mfc=light2,mec=dark2,marker='s',ms=3*linewidth,mew=linewidth,label='Training data')

  
  ax.set_xlabel('x',fontsize=fontsize,fontweight='bold')
  ax.set_ylabel('z',fontsize=fontsize,fontweight='bold') 
  ax.axis(axis_lim)
  ax.yaxis.grid(True)

  plt.legend(loc=1, ncol=3)#, mode="expand", borderaxespad=0.)

def shadow(x,y,**kwargs):  
  #color palette
  light1 =  Tango.colors('lightBlue')
  dark1  =  Tango.colors('darkBlue')
 
  #args
  expected_kwargs = ('linewidth','axis_lim','figsize','ref','data','fontsize','capthick','subplot')
  default_values  = (3,(0,10,-2,2),(24,9),None,None,22,6)
  linewidth,axis_lim,figsize,ref,data,fontsize,capthick = get_from_kwargs(expected_kwargs,default_values,kwargs)
  
  #fig    
  fig = plt.figure(figsize=figsize) 
  ax  = fig.add_subplot(1,1,1)
  

  ymean = np.mean(y,axis=0)
  plt.plot(x,ymean,color=dark1,linewidth=linewidth)
  
  for yi in y:
    plt.plot(x,yi,color=light1,linewidth=linewidth-2)
  
  ax.set_xlabel('Time step',fontsize=fontsize,fontweight='bold')
  ax.set_ylabel('Error[m]',fontsize=fontsize,fontweight='bold') 
  ax.axis(axis_lim)
  ax.yaxis.grid(True)

  plt.legend(loc=1, ncol=3)#, mode="expand", borderaxespad=0.)


def errorbar(x,y,yvar,**kwargs):
  #args
  x = x.copy().flatten()
  y = y.copy().flatten()
  yvar = yvar.copy().flatten()  
  expected_kwargs = ('linewidth','axis_lim','figsize','x_ref','y_ref','yvar_ref','fontsize','capthick','xlabel','ylabel')
  default_values  = (3,(0,10,-2,2),(24,9),None,None,None,22,6,'x','z')
  linewidth,axis_lim,figsize,x_ref,y_ref,yvar_ref,fontsize,capthick = get_from_kwargs(expected_kwargs,default_values,kwargs)
  
  light = Tango.colors('lightPurple')
  dark  = Tango.colors('darkPurple')

  #fig    
  fig = plt.figure(figsize=figsize) 
  ax  = fig.add_subplot(1,1,1)

  if (x_ref is not None) and (y_ref is not None) and (yvar_ref is not None):     
    y_ref_upper = y_ref + 2*(yvar_ref**0.5)
    y_ref_lower = y_ref - 2*(yvar_ref**0.5)
  
    refz,  = plt.plot(x_ref,y_ref,'--k',linewidth=linewidth,label='Reference function')
    refz2, = plt.plot(x_ref,y_ref_upper,':k',linewidth=linewidth,label='Reference confidence interval')
    plt.plot(x_ref,y_ref_lower,':k',linewidth=linewidth)
  
  plt.errorbar(x, y, yerr=2*(yvar**0.5), fmt='o',ecolor=light,capthick=capthick,elinewidth=linewidth,label='Training data',mfc=light,mec=dark,marker='s',ms=3*linewidth,mew=linewidth)
  ax.set_xlabel(xlabel,fontsize=fontsize,fontweight='bold')
  ax.set_ylabel(ylabel,fontsize=fontsize,fontweight='bold') 
  if axis_lim is not None:    
    ax.axis(axis_lim)
  ax.yaxis.grid(True)

  plt.legend(loc=1, ncol=3)#, mode="expand", borderaxespad=0.)


def surf(X,Y,**kwargs):
  #color palette
  light1 =  Tango.colors('lightBlue')
  light2 =  Tango.colors('lightRed')
  dark1  =  Tango.colors('darkBlue')
  dark2  =  Tango.colors('darkRed')
 
  #args

  linewidth = kwargs.get('linewidth',0.5)
  axis_lim  = kwargs.get('axis_lim',None)
  figsize   = kwargs.get('figsize',(16,12))
  fontsize  = kwargs.get('fontsize',22)
  capthick  = kwargs.get('capthick',6)
  subplot   = kwargs.get('subplot',None)
  ref       = kwargs.get('ref',None)
  data      = kwargs.get('data',None)

  #fig
  if subplot is None:    
    fig = plt.figure(figsize=figsize) 
    ax  = fig.add_subplot(111,projection='3d')
    # REFERENCE FUNCTION  
  else:
    ax = subplot  

  if ref is not None:
    ref_X = ref[0]
    ref_Y = ref[1]
    
    ax.plot_trisurf(ref_X[:,0],ref_X[:,1],ref_Y.copy().flatten(),color='k',alpha=0.1,linewidth=2*linewidth,edgecolors='k',label='Reference')


  ax.plot_trisurf(X[:,0],X[:,1],Y.copy().flatten(),color=light1,alpha=0.25,linewidth=linewidth,edgecolors=dark1,label='Prediction')

    
  if data is not None:
    data_X = data[0]
    data_Y = data[1]
    
    ax.scatter(data_X[:,0],data_X[:,1],data_Y.copy().flatten(),c=dark2,marker='s',s=50)

#    train = plt.errorbar(X, Y, yerr=2*(Yvar**0.5), fmt='o',ecolor=light2,capthick=capthick,elinewidth=linewidth,mfc=light2,mec=dark2,marker='s',ms=3*linewidth,mew=linewidth)

  
  ax.set_xlabel('x[m]',fontsize=fontsize,fontweight='bold')
  ax.set_ylabel('y[m]',fontsize=fontsize,fontweight='bold') 
  plt.legend(loc=1, ncol=3)#, mode="expand", borderaxespad=0.)


def plot(x,y,**kwargs):  
  #color palette
  light1 =  Tango.colors('lightBlue')
  light2 =  Tango.colors('lightPurple')
  light3 =  Tango.colors('lightRed')
  dark1  =  Tango.colors('darkBlue')
  dark2  =  Tango.colors('darkPurple')
  dark3  =  Tango.colors('darkRed')

 
  #args
  linewidth = kwargs.get('linewidth',3)
  axis_lim  = kwargs.get('axis_lim',(0,1,0,1))
  figsize   = kwargs.get('figsize',(12,4.5))
  fontsize  = kwargs.get('fontsize',22)
  capthick  = kwargs.get('capthick',6)
  subplot   = kwargs.get('subplot',None)
  ref       = kwargs.get('ref',None)
  yvar      = kwargs.get('yvar',None)
  data      = kwargs.get('data',None)
  data2     = kwargs.get('data2',None)  
  labels    = kwargs.get('labels',('','','','','','','','','','','','','','',''))

  #fig
  if subplot is None:    
    fig = plt.figure(figsize=figsize) 
    ax  = fig.add_subplot(1,1,1)
    # REFERENCE FUNCTION  
  else:
    ax = subplot  

  if ref is not None:
    y_ref = ref[0]
    yvar_ref = ref[1]     
    y_ref_upper = y_ref + 2*(yvar_ref**0.5)
    y_ref_lower = y_ref - 2*(yvar_ref**0.5)
  
    ax.plot(x,y_ref,'--k',linewidth=linewidth,label=labels[2])
    ax.plot(x,y_ref_upper,':k',linewidth=linewidth,label=labels[3])
    ax.plot(x,y_ref_lower,':k',linewidth=linewidth)
  #DATA1
  if data is not None:
    X = data[0].copy().flatten()
    Y = data[1].copy().flatten()
    if len(data) > 2:
      Yvar = data[2].copy().flatten()
    else:
      Yvar = 0
    ax.errorbar(X, Y, yerr=2*(Yvar**0.5),fmt='o',ecolor=light2,capthick=capthick, elinewidth=3,mfc=light2,mec=dark2,marker='s',ms=8,mew=3,label=labels[4])
  #DATA2
  if data2 is not None:
    X = data2[0].copy().flatten()
    Y = data2[1].copy().flatten()
    if len(data2) > 2:
      Yvar = data2[2].copy().flatten()
    else:
      Yvar = 0
    ax.errorbar(X, Y, yerr=2*(Yvar**0.5),fmt='o',ecolor=light3,capthick=capthick, elinewidth=3,mfc=light3,mec=dark3,marker='^',ms=8,mew=3,label=labels[5])
  
  # MAIN FUNCTION
  ax.plot(x,y,color=dark1,linewidth=linewidth,label=labels[0])
  if yvar is not None:
    y_upper = y + 2*(yvar**0.5) # +2 SD
    y_lower = y - 2*(yvar**0.5) # -2 SD
    ax.fill_between(x.flatten(),y_upper.flatten(),y_lower.flatten(),facecolor=light1,alpha=0.3)
    ax.plot([], [], light1,alpha=0.3,linewidth=10,label=labels[1])



  ax.set_xlabel('x',fontsize=fontsize,fontweight='bold')
  ax.set_ylabel('z',fontsize=fontsize,fontweight='bold') 
  ax.axis(axis_lim)
  ax.yaxis.grid(True)

  plt.legend(loc=1, ncol=3)#, mode="expand", borderaxespad=0.)

def contourf(X,Y,n=25):
  fig = plt.figure()
  ax1 = fig.add_subplot(1,1,1)
  size  = np.sqrt(np.prod(X[:,0].shape))
  shape = (size,size)
    
  C = ax1.contourf(np.reshape(X[:,0],shape),np.reshape(X[:,1],shape),np.reshape(Y,shape),n)
  CB = plt.colorbar(C, shrink=0.8, extend='both')
  plt.show()



