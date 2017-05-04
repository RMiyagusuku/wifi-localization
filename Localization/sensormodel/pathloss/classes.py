import numpy as np
import scipy
import scipy.stats as stats
import copy

from util import data_validation 


class PathLoss(object):
    """
    Base class for path loss functions. Each inherited class needs to implement 
    (1) mean_val, 
    (2) implement params_init, can be a fixed, educated random or random init.

    data:
        Dictionary with entries X, Y and Var
        X:      Location inputs X, numpy array [p,2]
        Y:      Readings at each position X, numpy array [p,n] or [p,]
        Var:    Readings variance, numpy array [p,n] or [p,]
    
    kwargs
        debug:      print debug statements - default 0
        params_0:   [n,np] numpy array, with np being the number of parameters of the 
                    particualr  model. It forces initial paramenters - default None
                    
    """
    def __init__(self,data,**kwargs):
        super(PathLoss,self).__init__()
        self.data    = data_validation.data_structure(data)
        self.debug   = kwargs.get('debug',0)
        self.params  = kwargs.get('params_0',None)
        
        if self.params is None:
            self.params = self.params_init()
        else:
            if self.params.ndim == 1:   #casting into [1,np]
                self.params = np.reshape(self.params,(1,self.params.shape[0])) 
            #Verify params_0 has adequate format
            assert params.shape[0] == data['Y'].shape[1]
        if self.debug:
            print 'class PathLoss init'        


    def nll(self,dataTest=None):
        """
        Negative log likelihood

        dataTest (Optional):
            If declared, will be used instead of self.data for computing the nll
        """
        
        if dataTest is None:
            data = self.data
        else:
            data  = data_validation.data_structure(dataTest)
    
        Ys  = self.mean_val(data['X']) #bounded estimation       
        log_prob = stats.norm.logpdf(Ys,loc=data['Y'],scale=data['Var']**0.5)
        return -np.sum(log_prob)
    

    def cv(self,dataTest=None):
        """
        Cross validation value used for comparing different function optimization outputs
        can be computed with a separate dataset test or with training dataset
        
        <math>
            nll(testing dataset)/nll(zero function)
        
        dataTest (Optional):
            If declared, will be used instead of self.data for computing the nll
        """

        if dataTest is None:
            data = self.data
        else:
            data  = data_validation.data_structure(dataTest)

        nll_f = self.nll(dataTest=data)
        nll_z = -np.sum(stats.norm.logpdf(np.zeros_like(data['Y']),loc=data['Y'],
                                                            scale=data['Var']**0.5))
        
        return nll_f/nll_z


    @staticmethod    
    def nll_optimize(params,*args): 
        """
        params:
            parameters used for the mean val function
        args:
            0:  Access Point 
            1:  mean val function
            2:  self.data
        """
        AP = args[0]        
        mean_val = args[1]
        X   = args[2]['X']
        Y   = args[2]['Y'][:,AP:AP+1]
        Var = args[2]['Var'][:,AP:AP+1]
        
        Ys  = mean_val(X,params=params)
        log_prob = stats.norm.logpdf(Ys,loc=Y,scale=Var**0.5)
        return -np.sum(log_prob)
    

    def optimize(self, AP='all', ntimes = 1, verbose=False):
        """
        Main optimization function
        Currently optimizes each AP independently, optimization is performed
        ntimes by optimizing nll using fmin_l_bfgs_b.

        AP(Optional):
            Is the access point to be optimized, if 'all' (default) all access
            points are optimized
        ntimes(Optional):
            number of optimizations performed, each with a random initial point
            given by params_init() - default 5
        verbose(Optional):
            If True outputs verbose information - default False
        """
        if AP == 'all':
            AP = range(self.data['Y'].shape[1])

        #Generating initial parameters
        #p0 = list()
        #p0.append(self.params)      #first parameters are current parameters                    
        #for i in range(1,ntimes):    
        #    p0.append(self.params_init())   #generates new set of initial params
        
        p0 = list()
        pinit = self.params_init()  #first parameters are current parameters                    
        p0.append(pinit)
        for theta in np.linspace(0,2*np.pi,ntimes-1,endpoint=False):           
            pi = pinit
            pi[:,0] += np.cos(theta)
            pi[:,1] += np.sin(theta)
            p0.append(pi)   #generates new set of initial params
        

        #Initiating optimization
        if verbose:
            print 'Initializing optimization'
        
        nll_values = np.inf*np.ones(self.data['Y'].shape[1])  #eval vector   

        for ntime in range(ntimes):
            if verbose:
                print 'Optimization ', ntime 
            #For each access points 
            for ap in AP:
                if verbose:
                    print 'Access Point: ', ap
                    #print self.params[ap:ap+1]
            
                args = (ap,self.mean_val,self.data)
                params0 = p0[ntime][ap:ap+1,:] #initial parameter from p0
                x,f,d = scipy.optimize.fmin_l_bfgs_b(self.nll_optimize, params0,
                                               args=args,approx_grad=True) #optimization alg
                #x,f,fc,gc,w = scipy.optimize.fmin_cg(self.nll_optimize, params0, 
                #                                 args=args,full_output=True)#, 
                           

                #if verbose:
                    #print params0                    
                    #print 'd: ', d

                if f < nll_values[ap]:   #if the optimization value is lower
                    if verbose:
                        print '########## Param vector updated ##########'
                        print x    
                        print nll_values[ap], ' to ', f 
                    nll_values[ap] = f              # update eval vector
                    self.params[ap,:] = x           # update params

    def new_data(self):
        """
        Ouputs a new dictionary where entry 'Y' has been replace by the difference between
        the original data entry and the bounded predicted value from mean_val()

        <math>
        Y_new = Y - mean_val(X)
        """

        data_new = copy.deepcopy(self.data)
        Ys = self.mean_val(self.data['X'])
        data_new['Y'] = data_new['Y']-Ys
        
        return data_new

    
    def __str__(self):
        """
        Overloading __str__ to make print statement meaningful
        
        <format>
        Name                    :
        data               
            X                   :
            Y                   :
            Var                 :
        Log-likelihood          :
        Number of Parameters    :
        Parameters
          ... all parameters and their values ...        
        """
        to_print = '{}  : {}\n'.format('Name'.ljust(32),self.name)
        to_print = to_print + 'data\n'
        to_print = to_print + '  {}  : [{},{}]\n'.format('X'.ljust(30),
                        str(self.data['X'].shape[0]).rjust(4),
                        str(self.data['X'].shape[1]).rjust(4))
        to_print = to_print + '  {}  : [{},{}]\n'.format('Y'.ljust(30),
                        str(self.data['Y'].shape[0]).rjust(4),
                        str(self.data['Y'].shape[1]).rjust(4))
        to_print = to_print + '  {}  : [{},{}]\n'.format('Var'.ljust(30),
                        str(self.data['Var'].shape[0]).rjust(4),
                        str(self.data['Var'].shape[1]).rjust(4))
        to_print = to_print + '{}  : {}\n'.format('Log-likelihood'.ljust(32),
                        str(self.nll()))
        to_print = to_print + '{}  : {}\n'.format('Number of Parameters'.ljust(32),
                        str(np.prod(self.params.shape)))
        ### Parameters
        to_print = to_print + 'Parameters\n'
        nap, nps = self.params.shape
        for ap,param in zip(range(nap),self.params):
            to_print = to_print + '{}:'.format(str(ap).rjust(4)[:4])
            for i in range(nps):
                to_print = to_print + ' {:16.6f}'.format(param[i])
            to_print = to_print + '\n'

        return to_print

    
    ####################          To implement by derived classes          ####################
    def mean_val(self,X,params=None,bounded=1):# params=None,i=0):
        """
        Compute the estimated value of the function at points X

        X:
            Points where the pathloss function is calculated
        bounded:
            if True, outputs the pathloss function bounded bellow by zero
            if False, outputs an unbounded value            
        params(Optional):
            For calculating mean_val. If None, use self.params. 
            Optional params important for optimization - see nll_optimize
        """
        raise NotImplementedError   

    def params_init(self):
        """
        Set params to suitable values at initialization if no init params (params_0) are given
        Outputs fixed, educated random or random initial parameters for optimization 
        """
        raise NotImplementedError
    
    
        

