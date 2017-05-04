import numpy as np

class MCL(object):
    """
    Base class for MCL.
    """
    def __init__(self,sensor_model,motion_model,**kwargs):
        super(MCLcore,self).__init__()
        self.name = 'MCLcore'
        assert isinstance(sensor_model,SensorModel)
        assert isinstance(motion_model,MotionModel)
        self.sensor_model   = sensor_model
        self.motion_model   = motion_model

        self.nsamples       = kwargs.pop('nsamples',800)
        self.particles_dim  = kwargs.pop('particles_dim',3)
        self.particles      = kwargs.pop('particles',np.random.rand(nsamples,particles_dim))

        #cumulative
        self.log = {'errors':[],'estimates':[]}

    def __str__(self):
        """
        Overloading __str__ to make print statement meaningful
        
        <format>
        Name                        :
        Number of samples           :
        Sensor model                :
        Motion model                :
        """
        to_print = '{}  : {}\n'.format('Name'.ljust(34),self.name)
        to_print = to_print + '{}  : {}\n'.format('Name'.ljust(34),self.pathloss.name)
        to_print = to_print + '{}  : {}\n'.format('Number of samples'.ljust(34),self.nsamples)
        to_print = to_print + '{}  : {}\n'.format('Sensor model'.ljust(34),self.sensor_model.name)
        to_print = to_print + '{}  : {}\n'.format('Motion model'.ljust(34),self.motion_model.name)
        return to_print

    def run(self,measurements,actions):
        """
        runs MCL over sensor measurements and motion actions
        saves errors, estimates and more at dict.log for each data pair
        """
        raise NotImplementedError

    


