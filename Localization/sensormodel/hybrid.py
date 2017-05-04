import GPy
from util import data_validation 
#from .pathloss.models import *
#from .pathloss.classes import PathLoss
from .pathloss.classes_c import FLog
from .gp import GPcore
import numpy as np

_eps = 1e-2

class hGP(GPcore):
    """
    Derived class from GPcore. Implements a hybrid pathloss-fingerprinting technique.
    Additionally uses for its implementation a pathloss.classes.PathLoss class
    """
    def __init__(self,data,**kwargs):
        super(hGP,self).__init__(data,**kwargs)
        self.name = 'GP with pathloss'        
        self.pathloss = kwargs.pop('pathloss',FLog(self.data))
        #raises error if pathloss is not a derived class of PathLoss
        #assert isinstance(self.pathloss,PathLoss)  
        
        self.nsd = kwargs.get('nsd',2.) #Number of standard deviations used to recalculate pathloss variance, 3 = 99.5%, 2 = 95%

        self.ndata = None
        self.gp    = '\n-> Run optimize()'

        if self.debug:
            print 'class hGP init works'
    
    def optimize(self,**kwargs):
        AP = kwargs.pop('AP','all')
        ntimes = kwargs.pop('ntimes',1)
        self.pathloss.optimize(AP=AP,ntimes=ntimes,verbose=self.verbose)
        self.ndata = self.pathloss.new_data()

        self.gp = GPy.models.GPRegression(self.ndata['X'],self.ndata['Y'],**kwargs)
        self.gp.optimize()

    def predict(self,X,**kwargs):
        flag_pl = kwargs.get('pl',False)    #returns path loss component
        flag_gp = kwargs.get('gp',False)    #returns GP component

        Y_gp,Yvar_gp = self.gp.predict(X)
        if flag_gp:
            return Y_gp, Yvar_gp

        Y_pl = self.pathloss.mean_val(X,bounded=False)
        if flag_pl:
            return Y_pl
        
        Ymean = Y_gp+Y_pl
        Ysd   = (1./self.nsd)*np.clip(self.nsd*Yvar_gp**0.5+Ymean-  
                    np.clip(Ymean,0,np.inf),_eps,np.inf)
        
        return np.clip(Ymean,0,np.inf), Ysd**2 + self.add_noise_var

    def __str__(self):
        """
        Overloading __str__ to make print statement meaningful
        
        <format>
        Name                        :
        data               
            X                       :
            Y                       :
            Var                     :
        
        PathLoss            
            Name                    :
            Log-likelihood          :
            Number of parameters    :

        Gaussian Process
            --GPy print
        
        """
        to_print = '{}  : {}\n'.format('Name'.ljust(34),self.name)
        to_print = to_print + 'data\n'
        to_print = to_print + '  {}  : [{},{}]\n'.format('X'.ljust(32),
                        str(self.data['X'].shape[0]).rjust(4),
                        str(self.data['X'].shape[1]).rjust(4))
        to_print = to_print + '  {}  : [{},{}]\n'.format('Y'.ljust(32),
                        str(self.data['Y'].shape[0]).rjust(4),
                        str(self.data['Y'].shape[1]).rjust(4))
        to_print = to_print + '  {}  : [{},{}]\n'.format('Var'.ljust(32),
                        str(self.data['Var'].shape[0]).rjust(4),
                        str(self.data['Var'].shape[1]).rjust(4))
        to_print = to_print + '\nPathLoss\n'
        to_print = to_print + '  {}  : {}\n'.format('Name'.ljust(32),self.pathloss.name)
        to_print = to_print + '  {}  : {}\n'.format('Log-likelihood'.ljust(32),
                        str(self.pathloss.nll()))
        to_print = to_print + '  {}  : {}\n'.format('Number of Parameters'.ljust(32),
                        str(np.prod(self.pathloss.params.shape)))
        to_print = to_print + '\nGaussian Process'
        to_print = to_print + self.gp.__str__().replace('\n','\n  ')       

        return to_print




