import numpy as np
from util.others import mesh
from scipy.stats import gaussian_kde

def resampling_wheel(particles,weights,Nsamples=None):
        '''
        Create new particles given particles' weights        
        Resampling wheel code addapted from Udacity's course 'Artificial Intelligence for Robotics'        
        '''
        N,dim = particles.shape
        if Nsamples is None:
            Nsamples = N
        p = np.empty((1,dim))


        index = np.random.randint(N)
        beta = 0.0
        mw = np.max(weights)
        for i in range(Nsamples+1):
            beta = beta + np.random.rand()*2*mw
            while beta > weights[index]:
                beta -= weights[index]
                index = (index+1)%N
            p = np.append(p, np.reshape(particles[index,:],(1,dim)), axis=0)
        return p[1:Nsamples+1,:]

def accept_reject_uniform(fun, *args, **kwargs):
    """
    accept-reject algorithm sampling using an uniform envelope distribution g(x)=M*uniform
    fun(x,*args) is the function to sample
    *args is the arguments for fun
    **kwargs
    max_fun is M
    dim is the input dimension for fun, input must be re-scaled to [0-1]
    dim2_0 is a flag that if true and dim=2 makes second dimension of x equal to zero - used when collapsing 2D data to 1D
    nsamples is the desired number of samples to be generated
    u_min = min value for u, so fun(x)/M must be always higher than u_min for it to be accepted  
    span is the area from where to sample
    batch is the factor of the 'batch', algorithm evaluates fun in batches of 'npoints' for speed
    """
    #TODO: use arbitrary g(x)
    
    max_fun  = kwargs.get('max_fun',1.)
    dim      = kwargs.get('dim',2)
    nsamples = kwargs.get('nsamples',800)
    u_min    = kwargs.get('u_min',0.)
    dim2_0   = kwargs.get('dim2_0',False)    
    min_max  = kwargs.get('span',(-20,80,-20,80))
    batch    = kwargs.get('batch',20)    
    npoints  = batch*nsamples
    samples  = np.empty((1,dim))
    

    _min_x     = min_max[0]*1.0
    _min_y     = min_max[2]*1.0
    _span_x    = min_max[1]*1.0-min_max[0]*1.0 
    _span_y    = min_max[3]*1.0-min_max[2]*1.0
    rep = 0
    while len(samples) < nsamples+1:
        x = np.random.rand(npoints,dim)
        x[:,0] = x[:,0]*_span_x+_min_x   
        x[:,1] = x[:,1]*_span_y+_min_y

        if dim == 2 and dim2_0:
            x[:,1] = x[:,1]*0
        px = fun(x, *args)
        u = np.random.rand(npoints)*(1.-u_min)+u_min
        samples = np.append(samples,x[u<(px/max_fun),:],axis=0)
        rep = rep+1
        max_fun = 0.9*max_fun
    return samples[1:nsamples+1,:], rep, len(samples)


def accept_reject_by_regions(fun, *args, **kwargs):
    """
    accept-reject algorithm sampling using an uniform by region envelope distribution 
    g(x)=M*uniformbyregion, the algorithm first divides the sampling region in KxK regions and generates 
    l_{k,k} uniform samples by each region with l_{k,k} being the relative probability of the region
 
    fun(x,*args) is the function to sample
    *args is the arguments for fun
    **kwargs
    max_fun is M
    dim is the input dimension for fun, input must be re-scaled to [0-1]
    dim2_0 is a flag that if true and dim=2 makes second dimension of x equal to zero - used when collapsing 2D data to 1D
    nsamples is the desired number of samples to be generated
    u_min = min value for u, so fun(x)/M must be always higher than u_min for it to be accepted  
    span is the area from where to sample
    batch is the factor of the 'batch', algorithm evaluates fun in batches of 'npoints' for speed
    K is the number of regions
    """
    
    max_fun  = kwargs.get('max_fun',2.5)
    dim      = kwargs.get('dim',2)
    nsamples = kwargs.get('nsamples',800)
    u_min    = kwargs.get('u_min',0.)
    dim2_0   = kwargs.get('dim2_0',False)    
    min_max  = kwargs.get('span',(-20,80,-20,80))
    batch    = kwargs.get('batch',1)    
    K        = kwargs.get('K',15)    

    npoints  = batch*nsamples
    samples  = np.empty((1,dim))    
    #psamples = np.empty(1)
    
    if len(min_max)==2:
        min_max = (min_max[0],min_max[1],min_max[0],min_max[1])    

    _min_x     = min_max[0]
    _min_y     = min_max[2]
    _span_x    = min_max[1]-min_max[0] 
    _span_y    = min_max[3]-min_max[2]


    #KxK regions
    _minmax_x = np.linspace(0,_span_x,K,endpoint=False)+_min_x
    _minmax_y = np.linspace(0,_span_y,K,endpoint=False)+_min_y
    _minmax = mesh(_minmax_x,_minmax_y)
    
    #Eval p() of each region
    _eval_x = np.linspace(0,_span_x,K+1)+_min_x  #corners of each region
    _eval_y = np.linspace(0,_span_y,K+1)+_min_y  #corners of each region
    evalx = mesh(_eval_x,_eval_y)
    
    pc = fun(evalx,*args,**kwargs)     #p() of the corner of each region
    
    PX = np.reshape(pc,(K+1,K+1)) # matrix form
    px = np.zeros((K,K)) # the p() of each region is calculated as the average of the corners  
    for j in range(K):
        for k in range(K):
            px[j,k] = np.sum(PX[j:j+2,k:k+2])/4
    px = px.flatten()    
    _px = np.round(npoints*px/np.sum(px))
    min_px = np.min(px[_px>=0])
    
    px = np.clip(px, min_px,np.inf)
    max_funi = max_fun*px
    
    vector = np.concatenate((_minmax,max_funi[:,None]),axis=1)
    
    #Accept-reject base algorithm
    rep = 0
    while len(samples) < nsamples+1:
        #sampling grid space to sample from using px
        sampled_vector = resampling_wheel(vector,px,Nsamples=npoints)
        xs = np.random.rand(npoints,dim)
        xs[:,0] *= _span_x/K   
        xs[:,1] *= _span_y/K
        xs += sampled_vector[:,0:2]

        if dim == 2 and dim2_0:
            xs[:,1] = xs[:,1]*0
        pxs = fun(xs, *args,**kwargs)
        u = np.random.rand(npoints)
        samples   = np.append(samples,xs[u<(pxs/sampled_vector[:,2]),:],axis=0)
        #psamples  = np.append(psamples,pxs[u<(pxs/sampled_vector[:,2]),:])
        rep = rep+1
        #max_fun = 0.9*max_fun
    return samples[1:nsamples+1,:], _px, max_funi, max_fun, rep


def accept_reject_by_regions_map(fun, *args, **kwargs):
    max_fun  = kwargs.get('max_fun',1.25)
    dim      = kwargs.get('dim',2)
    nsamples = kwargs.get('nsamples',800)
    u_min    = kwargs.get('u_min',0.)
    dim2_0   = kwargs.get('dim2_0',False)    
    min_max  = kwargs.get('span',(-20,80,-20,80))
    batch    = kwargs.get('batch',1)    
    K        = kwargs.get('K',15)    
    lrfmap   = kwargs.get('lrfmap',None)
    
    npoints  = batch*nsamples
    if lrfmap is not None:
        # several samples are expected to be rejected, so an additional factor is introduced
        npoints = 3*npoints
    samples  = np.empty((1,dim))    
   

    if len(min_max)==2:
        min_max = (min_max[0]*1.0,min_max[1]*1.0,min_max[0]*1.0,min_max[1]*1.0)    

    _min_x     = min_max[0]*1.0
    _min_y     = min_max[2]*1.0
    _span_x    = min_max[1]*1.0-min_max[0]*1.0 
    _span_y    = min_max[3]*1.0-min_max[2]*1.0


    #KxK regions
    _minmax_x = np.linspace(0,_span_x,K,endpoint=False)+_min_x
    _minmax_y = np.linspace(0,_span_y,K,endpoint=False)+_min_y
    _minmax = mesh(_minmax_x,_minmax_y)
    
    #Eval p() of each region
    _eval_x = np.linspace(0,_span_x,K+1)+_min_x  #corners of each region
    _eval_y = np.linspace(0,_span_y,K+1)+_min_y  #corners of each region
    evalx = mesh(_eval_x,_eval_y)
    
    pc = fun(evalx,*args,**kwargs)     #p() of the corner of each region
    
    PX = np.reshape(pc,(K+1,K+1)) # matrix form
    px = np.zeros((K,K)) # the p() of each region is calculated as the average of the corners  
    for j in range(K):
        for k in range(K):
            px[j,k] = np.sum(PX[j:j+2,k:k+2])/4.0
    px = px.flatten()    
    _px = np.round(npoints*px/np.sum(px))
    min_px = np.min(px[_px>=0])
    
    px = np.clip(px, min_px,np.inf)
    max_funi = max_fun*px
    
    vector = np.concatenate((_minmax,max_funi[:,None]),axis=1)
    
    #Accept-reject base algorithm
    rep = 0
    while len(samples) < nsamples+1:
        #sampling grid space to sample from using px
        sampled_vector = resampling_wheel(vector,px,Nsamples=npoints)
        xs = np.random.rand(npoints,dim)
        xs[:,0] *= _span_x/K   
        xs[:,1] *= _span_y/K
        xs += sampled_vector[:,0:2]

        if dim == 2 and dim2_0:
            xs[:,1] = xs[:,1]*0
           
        #check map occupancy
        if lrfmap is not None:
            free_xs = lrfmap.occupancy_free(xs)
            xs = xs[free_xs,:]
            sampled_vector = sampled_vector[free_xs,:]
        
        pxs = fun(xs, *args,**kwargs)
        u = np.random.rand(xs.shape[0])
        
        samples   = np.append(samples,xs[u<(pxs/sampled_vector[:,2]),:],axis=0)
        #psamples  = np.append(psamples,pxs[u<(pxs/sampled_vector[:,2]),:])
        rep = rep+1
        #max_fun = 0.9*max_fun
    return samples[1:nsamples+1,:], _px, max_funi, max_fun, rep


