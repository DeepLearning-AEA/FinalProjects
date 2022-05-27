class Loss(object):
    '''Base class for all loss functions'''
    def forward(self, y_pred, y_true):
        '''Given a prediction and the ground truth, computes the loss'''
        raise NotImplementedError
        
    def compute_gradient(self):
        '''Given the previous input, computes the gradient of the loss'''
        raise NotImplementedError
        
class MSELoss(Loss):
    '''
    The Mean-Squared Error loss
    '''
    def __init__(self):
        self.difference = None
        self.n = None
        
    def forward(self, y_pred, y_true):
        """
        Compute the loss 
        """
        assert y_pred.shape == y_true.shape
        self.difference = y_pred - y_true
        self.n = sum([p**2 for p in y_pred.shape])
        return (self.difference.pow(2)).mean()
    
    def compute_gradient(self):
        """
        The output of the backward pass of a loss function is simply its derivative.
        """
        assert self.difference is not None, 'Forward pass must happen before backward pass'
        return 2*self.difference/self.n
    
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)