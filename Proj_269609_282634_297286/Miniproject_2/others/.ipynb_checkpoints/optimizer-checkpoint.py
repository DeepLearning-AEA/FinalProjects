from torch import Tensor
from torch import empty

class Optimiser(object):
    '''Base class for all optimisers'''
    def __init__(self, parameters, lr):
        '''Input : list of parameters to optimize, learning rate'''
        self.lr = lr
        self.parameters = []
        # Flatten the list of parameters
        for p in parameters:
            self.parameters.extend(p)
        
    def step(self):
        '''Does one step of optimization'''
        raise NotImplementedError
        
        
class SGD(Optimiser):
    '''Stochastic Gradient Descent (SGD) Optimiser'''
    def step(self):
        """
        Perform one step of otpimization
        """
        for p in self.parameters:
            # Update all the paramters by -learning_rate*gradients
            p[0].sub_(self.lr*p[1])
                    
class Adam(Optimiser):
    """
    Adam Optimizer
    """
    def __init__(self, parameters, lr, betas=(0.9, 0.99)):
        super().__init__(parameters, lr)
        # Dictionnaries that will contain the estimate of the mean and the "speed" of the optimizer
        self.mean_estim = {}
        self.speed_estim = {}
        
        # Betas parameters for the adam optimizer
        self.betas = betas
        
        # Initialise all the parameters estimate to 0 and with the correct shape
        for (idx,p) in enumerate(self.parameters):
            self.mean_estim[idx] = empty(p[1].shape).zero_().to(p[1].device)
            self.speed_estim[idx] = empty(p[1].shape).zero_().to(p[1].device)
            
        # Set the step counter and the epsilon factor 
        self.eps = 1e-8
        self.step_count = 0
    
    def step(self):
        # Increase step counter
        self.step_count += 1
        # Get the ADAM paramter
        beta1, beta2 = self.betas
        for (idx,p) in enumerate(self.parameters):
            # For current parameters, get the current estimate of the mean and the "speed"
            cur_mean = self.mean_estim[idx]
            cur_speed = self.speed_estim[idx]
            
            # Update it according to ADAM optimizer formula
            cur_mean = beta1*cur_mean+(1-beta1)*p[1]
            cur_speed = beta2*cur_speed +(1-beta2)*(p[1]**2)
            
            # Save the new vaues
            self.mean_estim[idx] = cur_mean
            self.speed_estim[idx]= cur_speed
            
            # Compute the hat-estimate for the update
            hat_cur_mean = cur_mean/(1-beta1**self.step_count)
            hat_speed_mean = cur_speed/(1-beta2**self.step_count)
            
            # update the parameters
            p[0].sub_(self.lr*hat_cur_mean/((hat_speed_mean**(1/2))+self.eps))
                