import numpy as np
from .motionmodel import MotionModel

_eps = 1e-10

class Odometry(MotionModel):
    """
    Given odometry information. It considers any motion can be generated from three basic
    motions: a rotation, a translation and a second rotation - and computes the motion based on
    this assumption.
    """

    def __init__(self,**kwargs):
        super(Odometry,self).__init__()
        self.name = 'Odometry'

        self.angle_var       = kwargs.get('angle_var',0.25)
        self.distance_var    = kwargs.get('distance_var',0.5)

    def forward(self,particles,odometry,**kwargs):
        """
        Forward motion using the odometry model. 
    
        particles:      
            Particles to be propagated, a [1x3] or a  [NX3] numpy array.
        Odometry:
            Odometry information in x,y and heading angle w.r.t. x at time t-1 and t, a
            [2x3] numpy array
        
        
        out: 
            new particles generated, a [Nx3] numpy array

        Example
            p_next = motionmodel.forward(particles, dataTest['Pose'][i-1:i+1,:])

        """       
        assert particles.shape[1] == 3

        N = kwargs.get('N',particles.shape[0])

        Pose_prev = odometry[0]
        Pose_next = odometry[1]

        delta_x = Pose_next[0]-Pose_prev[0]
        delta_y = Pose_next[1]-Pose_prev[1]
        delta_t = Pose_next[2]-Pose_prev[2]

        x_0 = particles[:,0]
        y_0 = particles[:,1]
        t_0 = particles[:,2]

        # Noise parameters    
        alpha1 = self.angle_var + _eps
        alpha2 = self.distance_var*self.angle_var + _eps**2
        alpha3 = self.distance_var + _eps
        alpha4 = self.distance_var*self.angle_var + _eps**2

        # Decomposing odometry into three motions
        d_rot1  = np.arctan2(delta_y,delta_x)-Pose_prev[2]
        d_trans = (delta_x**2+delta_y**2)**0.5
        d_rot2  = delta_t-d_rot1

        _d_rot1 = np.abs(d_rot1)
        _d_trans= np.abs(d_trans)
        _d_rot2 = np.abs(d_rot2)

        d_rot1_n = d_rot1+np.random.normal(size=N,
                            scale=_eps+(alpha1*_d_rot1+alpha2*_d_trans)**0.5)
        d_trans_n= d_trans+np.random.normal(size=N,
                            scale=_eps+(alpha3*_d_trans+alpha4*_d_rot1+alpha4*_d_rot2)**0.5)
        d_rot2_n = d_rot2 + np.random.normal(size=N,
                            scale=_eps+(alpha1*_d_rot2+alpha2*_d_trans)**0.5)

        x_n = x_0 + d_trans_n*np.cos(t_0+d_rot1_n)
        y_n = y_0 + d_trans_n*np.sin(t_0+d_rot1_n)
        theta_n = t_0 + d_rot1_n +d_rot2_n
   
        P = np.asarray([x_n,y_n,theta_n])
    
        return P.T


    def backward(self,particles,odometry,**kwargs):
        """
        Backward motion using the odometry model. It considers odometry information is generated
        from three motions: a rotation, a translation and a second rotation. For backward motion 
        the most likely particle is computed, hence variances are scaled  by a factor of 1/10
   
        particles:      
            Particles to be propagated, a [1X3] or a [NX3] numpy array
        Odometry:
            Odometry information in x,y and heading angle w.r.t. x at time t-1 and t, a
            [2x3] numpy array
        
        outputs:
            array [Nx3] of particles generated

        Example:     
        p_prev = model.odometry_P3_back(dataTest['Pose'][i,:],
                    dataTest['Pose'][i-1,:],particles_next=p_next,angle_var=0.00025,
                    distance_var=0.0005,N=N)
        """
        assert particles.shape[1] == 3

        N = kwargs.get('N',particles.shape[0])

        Pose_prev = odometry[0]
        Pose_next = odometry[1]

        delta_x = Pose_next[0]-Pose_prev[0]
        delta_y = Pose_next[1]-Pose_prev[1]
        delta_t = Pose_next[2]-Pose_prev[2]

        x_1     = particles[:,0]
        y_1     = particles[:,1]
        t_1     = particles[:,2]

        # Noise parameters 
        alpha1  = (1/10)*self.angle_var + _eps
        alpha2  = (1/100)*self.distance_var*self.angle_var + _eps**2
        alpha3  = (1/10)*self.distance_var + _eps
        alpha4  = (1/100)*self.distance_var*self.angle_var + _eps**2

        # Decomposing odometry into three motions
        d_rot2  = Pose_next[2]-np.arctan2(delta_y,delta_x)
        d_trans = (delta_x**2+delta_y**2)**0.5
        d_rot1  = delta_t-d_rot2

        _d_rot1 = np.abs((d_rot1+np.pi)%(2*np.pi)-np.pi)
        _d_trans= d_trans
        _d_rot2 = np.abs((d_rot2+np.pi)%(2*np.pi)-np.pi)

        d_rot1_n = d_rot1+np.random.normal(size=N,
                                scale=_eps+(alpha1*_d_rot1+alpha2*_d_trans)**0.5)
        d_trans_n= d_trans+np.random.normal(size=N,
                                scale=_eps+(alpha3*_d_trans+alpha4*_d_rot1+alpha4*_d_rot2)**0.5)
        d_rot2_n = d_rot2 + np.random.normal(size=N,
                                scale=_eps+(alpha1*_d_rot2+alpha2*_d_trans)**0.5)

        x_0 = x_1 - d_trans_n*np.cos(t_1-d_rot2_n)
        y_0 = y_1 - d_trans_n*np.sin(t_1-d_rot2_n)
        t_0 = t_1 - d_rot1_n - d_rot2_n
   
        P = np.asarray([x_0,y_0,t_0])
    
        return P.T


