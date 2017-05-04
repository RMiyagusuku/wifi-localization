import numpy as np
import scipy
import scipy.stats as stats
import copy
import levmar
from util import data_validation 
from util.distance import euclidean_squared


class FLog(object):
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
        self.data    = data_validation.data_structure(data)
        self.debug   = kwargs.get('debug',0)
        self.name    = 'Log Function with levmar optimization'
        #self.params  = kwargs.get('params_0',None)
        
        self.params = np.ones((self.data['Y'].shape[1],4))

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

    def optimize(self, AP='all', ntimes = 1, verbose=False):
        """
        Main optimization function
        Call for levmar optimization 
        TODO: add openMP to levmar
        """
        # Optimization through levmar
        if verbose:
            print 'Initializing optimization'

        nap = self.data['Y'].shape[1]
        x_list = self.data['X'][:,0].tolist()
        y_list = self.data['X'][:,1].tolist()
        z_list = self.data['Y'].T.flatten().tolist()
        p_estimate = levmar.optimize(x_list,y_list,z_list)
        self.params = np.reshape(np.asarray(p_estimate),self.params.shape)
        

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
    def mean_val(self,X,params=None,bounded=True):
        """
        Compute the estimated value of the function at points X - see .classes.PathLossFun

        <math>
            Pr = P0 - k*log10(|Xap-X|)
        """

        if params is None:
            params = self.params

        XAP = params[:,2:4]
        k  = params[:,1]
        P0 = params[:,0]    

        r2 = euclidean_squared(X,XAP)
        out = P0 - k*np.log(r2)

        if bounded:
            return np.clip(out,0.,P0)
        else:
            return np.clip(out,-np.inf,P0)     

