import numpy as np

class MotionModel(object):
    def __init__(self):
        super(MotionModel,self).__init__()
        self.name = 'Motion model'

    def __str__(self):
        to_print = self.name
        to_print = to_print + 'all params'


    def forward(self,particles,odometry,**kwargs):
        """
        Forward motion
        """
        raise NotImplementedError


    def backward(self,particles,odometry,**kwargs):
        """
        Backward motion
        """
        raise NotImplementedError


    def __str__(self):
        """
        Overloading __str__ to make print statement meaningful
        
        <format>
        Name                        :
        -- all params --        
        """
        to_print = '{}  : {}\n'.format('Name'.ljust(34),self.name)
        to_print = to_print + 'Parameters'
        for key,value in self.__dict__.items():
            if key!='name':
                to_print = to_print + '\n  {}  : {}'.format(key.ljust(32),value)
        return to_print
