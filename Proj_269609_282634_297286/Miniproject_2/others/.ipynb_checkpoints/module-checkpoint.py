import math
from torch import empty , cat , arange
from torch.nn.functional import fold, unfold

def he_init(tensor, n_in):
    """
    Initialise the given tensor with the given standard deviation (according to He rule)
    :param tensor: the tensor to initialise 
    :param n_in: the number of parameters of the layer
    """
    tensor.normal_(0, 2/n_in)

class Network():
    def __init__(self):
        """
        Construction of the requested network
        """
        self.in_channel = 3
        self.out_channel_1 = 48
        self.out_channel_2 = 48
        self.seq = Sequential([
                Conv2d(self.in_channel, self.out_channel_1, kernel_size =3, stride=2, padding=1),
                ReLU(),
                Conv2d(self.out_channel_1, self.out_channel_2, kernel_size =3, stride=2,padding=1), 
                ReLU(),
                Upsampling(self.out_channel_2, self.out_channel_1,kernel_size=3,padding=1), 
                ReLU(),
                Upsampling(self.out_channel_1, self.in_channel,kernel_size=3,padding=1),
                Sigmoid()
            ])
        # Initialise all the parameters of the convolution layer
        self.seq.init_parameters()
        
    def forward(self, input):
        """
        Forward the input through the whole model
        :param input: the input to pass through the network
        :return: the processed input
        """
        return self.seq(input)
    
    
    def __call__(self, input):
        # Override the __call__ function so that the model(x) computes the forward w.r.t. x
        return self.forward(input)

    def backward(self, grad_loss):
        """
        Compute the backward pass on the whole model with the given loss gradient 
        :param grad_loss: the gradient of the loss
        """
        self.seq.backward(grad_loss)
    
    def param(self):
        """
        Return the list of pairs of (parameters,gradients) of the model
        :return: the list of pairs (parameters, gradients) of the model
        """
        return self.seq.param()
    
    def zero_grad(self):
        """
        Set the gradient of the model w.r.t. all the parameters to 0
        """
        self.seq.zero_grad()
        
    def load(self, data):
        """
        Load the model parameters
        :param data: the list of paramters of the model
        """
        self.seq.load_param(data)
        
    def to(self, device):
        """
        Move the model to the specified device
        :param device: the device to which we should move the model
        """
        self.seq.to(device)
        
    
class Module(object) :
    """
    Base class for every module
    """
    def forward (self, input) :
        """
        Forward the input through the module
        :param input: the input to pass through the module
        :return: the processed input
        """
        raise NotImplementedError
    
    def backward(self, *gradwrtoutput):
        """
        Return the gradient with respect to the input of the module
        :param gradwrtoutpt: the gradient with respect to the output of the current module
        :return: the gradient with respect to the input of the module
        """
        raise NotImplementedError
            
    def param(self) :
        """
        a list of pairs composed of a parameter tensor and a gradient tensor
        """
        return []
    
    def zero_grad(self):
        """
        Set the gradient of the module w.r.t. all the parameters to 0
        """
        pass
    
    def init_parameters(self):
        """
        Initialise the parameters of the module
        """
        pass
    
    def to(self,device):
        """
        Move the parameters of the module to the specified device
        :param device: the device on which we should move the device
        """
        pass
    
    def __call__(self, input):
        # Override the call functon so that it calls the forward method instead
        return self.forward(input)
    
    def load_param(self, param):
        """
        Load the parameters in the module
        :param param: the parameters of the module
        """
        pass

    
class NNUpsampling(Module):
    """
    Module repreenting an upsampling of the input by a given integer factor. 
    """
    def __init__(self, scale_factor):
        """
        Construct the object
        :param scale_factor: the scale_factor used to upsample the images
        """
        super().__init__()
        self.scale_factor = scale_factor
        
    def forward(self, input):
        """
        Forward the input through the module
        :param input: the input to pass through the module
        :return: the processed input
        """
        assert input.dim() == 4, "The input tensor must be of the dimension 4 (NxCxHxW)"
        # Save the input for backward pass 
        self.input = input.clone()
        
        # Used the repeat_interleave function to perform upsampling of the function
        return input.repeat_interleave(self.scale_factor, dim=2).repeat_interleave(self.scale_factor,dim=3)
    
    def backward(self, *gradwrtoutput):
        """
        Return the gradient with respect to the input of the module
        :param gradwrtoutpt: the gradient with respect to the output of the current module
        :return: the gradient with respect to the input of the module
        """
        # Put the correct outputs from later summation
        # Sum over all output generated from one input
        # reshape to have the original shape
        return unfold(gradwrtoutput[0], kernel_size=self.scale_factor, stride=self.scale_factor)\
                    .reshape((self.input.shape[0], self.input.shape[2], self.scale_factor*self.scale_factor,  - 1))\
                    .sum(2)\
                    .reshape(self.input.shape)
        
class Sequential(Module):
    """
    A class that takes as inputs a list of ordered Module,  and considers them as sequential layers of a neural net.
    """
    def __init__(self, *modules):
        """
        Build a sequential layer
        :param modules: a list-like of neural network module that will be applied sequentially
        """
        super().__init__()
        self._modules = list(*modules)
    
    def apply_to_all_mod(self, lambda_mod):
        """
        Helper functions for the sequential module to apply a function to all modules in the sequential layer
        """
        for mod in self._modules:
            lambda_mod(mod)
            
    def forward(self, x):
        """
        Forward the input through the module
        :param x: the input to pass through the module
        :return: the processed input
        """
        # Apply sequentially the input to all the layers
        for mod in self._modules:
            x = mod(x)
        return x
    
    def backward(self, *gradwrtoutput):
        """
        Return the gradient with respect to the input of the module
        :param gradwrtoutpt: the gradient with respect to the output of the current module
        :return: the gradient with respect to the input of the module, i.e. the first layer of the sequential module
        """
        grad = gradwrtoutput[0]
        # Go through the layers in reversed order and compute the backward on all the modules
        for m in reversed(self._modules):
            grad = m.backward(grad)
            
        # Return the gradient w.r.t. input of the layer, i.e. the gradient w.r.t. to the input of the first layer in the sequential
        return grad
    
    def zero_grad(self):
        """
        Set the gradient of the module w.r.t. all the parameters to 0
        """
        # Zeroed the gradient of all the layers in the sequential
        self.apply_to_all_mod(lambda mod: mod.zero_grad())
    
    def init_parameters(self):
        """
        Initialise the parameters of all the module in the sequential layers
        """
        self.apply_to_all_mod(lambda mod: mod.init_parameters())
            
    def param(self):
        """
        Get the list of parameters, gradients pair of the convolution layer
        :return: the list of parameters of the module
        """
        all_params = []
        self.apply_to_all_mod(lambda mod: all_params.append(mod.param()))
        return all_params
    
    def to(self, device):
        """
        Move the parameters of the module to the specified device
        :param device: the device on which we should move the device
        """
        self.apply_to_all_mod(lambda mod: mod.to(device))
    
    def load_param(self, data):
        """
        Load the parameters in the different modules
        :param data: the parameter to load
        """
        
        for i, mod in enumerate(self._modules):
            mod.load_param(data[i])
        
class Upsampling(Sequential):
    """
    Class representating an upsampling operation, i.e. a combination of a nearest neighbour upsampling followed by a convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor=2,padding=0):
        """
        Build an upsampling layer
        :param in_channels: the number of channel as input of the convolution layer
        :param out_channels: the number of channel as output of the convolution layer
        :param kernel_size: the kernel_size of the convolution
        :param scale_factor: the scale_factor for the upsampling layer
        :param padding: the padding used in the convolution layer
        """
        assert type(scale_factor)==int, "The scale_factor parameter should be an integer"
        super().__init__([
            NNUpsampling(scale_factor),
            Conv2d(in_channels, out_channels, kernel_size,padding=padding)
        ])
        
    def param(self):
        """
        Get the list of parameters, gradients pair of the convolution layer
        :return: the list of parameters of the module
        """
        return self._modules[1].param()
    
    def load_param(self, data):
        """
        Load the parameter of the convolution inside the correct module
        :param data: the parameter of the convolution
        """
        self._modules[1].load_param(data)
        
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride =1, bias=True):
        """
        Initialise a convolution module with the given parameters
        :param in_channels: the number of input channels in the convolution
        :param out_channels: the number of output channels in the convolution
        :param kernel_size: int or tuple, the kernel size used for the convolution
        :param padding: the padding used for the convolution
        :param stride: the stride used for the convolution
        :param bias: true if should use the bias term, false otherwise
        """
        super().__init__()
        
        # transform parameters to tuple if they are int
        if(type(kernel_size) is not tuple):
            kernel_size = (kernel_size, kernel_size)
        if(type(padding) is not tuple):
            padding = (padding, padding)
        if(type(stride) is not tuple):
            stride = (stride, stride)
        
        # store parameters
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.weights = empty(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        self.bias = empty(self.out_channels) if bias else None
        self.padding = padding
        self.stride = stride
        
        # define the save input for the backward pass
        self.input_save = None
        
        # Define gradients
        self.gradB = empty(self.out_channels).zero_()
        self.gradW = empty((self.out_channels,self.in_channels, self.kernel_size[0], self.kernel_size[1])).zero_()
        
    def forward(self, x):
        """
        Forward the input through the module
        :param x: the input to pass through the module
        :return: the processed input
        """
        
        # Define the output size of the convolution
        w_out = math.floor((x.shape[2] + 2*self.padding[0] - (self.kernel_size[0]-1)-1)/self.stride[0]+1)
        h_out = math.floor((x.shape[3] + 2*self.padding[1] - (self.kernel_size[1]-1)-1)/self.stride[1]+1)
        
        # Compute the convolution
        x_unf = unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding).to(x.device)
        
        
        self.input_unfolded = x_unf.clone()
        self.input_size = x.shape
        
        wxb = self.weights.view(self.out_channels, -1)@ x_unf 
        if self.bias is not None:
            wxb+= self.bias.view(1, -1, 1)
            
        result = wxb.view(x.shape[0], self.out_channels, w_out, h_out)
        
        return result
    
    def zero_grad(self):
        """
        Set the gradient of the module w.r.t. all the parameters to 0
        """
        self.gradW.zero_()
        self.gradB.zero_()
    
    def init_parameters(self):
        """
        Initialise the parameters of the module according to 
        - He initialization for the weights
        - 0.01 for the bias (according to online search)
        """
        shape = self.weights.shape
        he_init(self.weights, shape[1]*shape[2]*shape[3])
        self.bias.fill_(0.01) #https://medium.com/@glenmeyerowitz/bias-initialization-in-a-neural-network-2e5d26fed0f0
    
    def to(self, device):
        """
        Move the model to the specified device
        :param device: the device to which we should move the model
        """
        self.bias = self.bias.to(device)
        self.gradB = self.gradB.to(device)
        self.weights = self.weights.to(device)
        self.gradW = self.gradW.to(device)
    
    def load_param(self, data):
        """
        Load the parameter of the convolution inside the correct module
        :param data: the parameter of the convolution
        """
        w, b = (data[0], data[1]) if self.bias is not None else (data, None)
        self.weights.zero_().add_(w[0])
        self.gradW.zero_().add_(w[1])
                                                                 
        if self.bias is not None:
            self.bias.zero_().add_(b[0])
            self.gradB.zero_().add_(b[1])
    
    def backward(self, *gradwrtoutput):
        """
        Return the gradient with respect to the input of the module
        :param gradwrtoutpt: the gradient with respect to the output of the current module
        :return: the gradient with respect to the input of the module
        """
        # The backward pass implementation has been implemented inspired by the following explanation 
        # https://coolgpu.github.io/coolgpu_blog/github/pages/2020/10/04/convolution.html
        assert self.input_unfolded is not None, "The forward pass must happen before the backward pass"
        gradwrtoutput = gradwrtoutput[0]
        
        input_shape_unf = self.input_unfolded.size()
        x_unf = self.input_unfolded
        input_size = self.input_size
        
        if self.bias is not None:
            # If we use bias, the update rule is simply the sum over all the output for which this bias contribute, i.e. the all the samples and all the outputs for one channel
            self.gradB.add_(gradwrtoutput.sum(axis=(0, 2, 3)))
        
        # Permute the gradient with respect to output to (Nout, out_1, out_2, N)  and reshape it as (N_out, out_1*out_2*N)
        gradwrtoutput_resh = gradwrtoutput.permute((1,2,3,0)).reshape(self.out_channels, -1)
        # Obtained the gradient of the weights with size (Nout, Nin*kernel_size_1*kernel_size_2) , according to the course formula
        # gradwrtoutput_resh is a (Nout, out_1*out_2*N), x_unf transposed from (N, Nin*kernel_size_1*kernel_size_2, out_1*out_2) to (N*out_1*out_2, N_in*kernel_size_1*kernel_size_2)
        gradW = gradwrtoutput_resh @ x_unf.permute((1,2,0)).reshape(input_shape_unf[1], input_shape_unf[0]*input_shape_unf[2]).T
        # Reshape it to the gradient size of (Nout, Nin, kernel_size_1, kernel_size_2)
        gradW = gradW.reshape(self.weights.shape)
        # Add it to the gradient field
        self.gradW.add_(gradW)
        
        # Reshape the weights to a (Nout, Nin*kernel_size_1*kernel_size_2) matrix
        gradwrtoutput_resh_2 = gradwrtoutput.view(x_unf.shape[0],self.out_channels, -1)
        grad_in = (self.weights.view(self.out_channels,-1).T @ gradwrtoutput_resh_2).view(input_shape_unf)
        
        # Use fold to reshape and sum the unfolded gradients to the correct shape for the input
        grad_X = fold(grad_in, (input_size[2], input_size[3]), kernel_size=self.kernel_size, stride =self.stride, padding = self.padding)
        
        return grad_X
        
    def param(self):
        """
        Get the list of parameters, gradients pair of the convolution layer
        :return: the list of parameters of the module
        """
        return [(self.weights, self.gradW)] + ([] if self.bias is None else [(self.bias, self.gradB)])
        
        
class ReLU(Module):
    def __init__(self):
        """
        Build the ReLU function
        """
        self._activation_ = None
    
    def forward(self, x):
        """
        Forward the input through the module
        :param x: the input to pass through the module
        :return: the processed input
        """
        self._activation_ = x.clone()
        return x.clamp(min = 0)
    
    def backward(self, *gradwrtoutput):
        """
        Return the gradient with respect to the input of the module
        :param gradwrtoutpt: the gradient with respect to the output of the current module
        :return: the gradient with respect to the input of the module
        """
        assert self._activation_ is not None, 'Forward pass must happen before backward pass'
        
        input_ = gradwrtoutput[0].clone()

        to_mult = self._activation_
        to_mult[to_mult >= 0] = 1
        to_mult[to_mult < 0] = 0

        return to_mult.mul(input_)
    
class LeakyReLU(Module):
    def __init__(self, alpha=0.1):
        """
        Build the ReLU function
        """
        self._activation_ = None
        self.alpha = alpha
    
    def forward(self, x):
        """
        Forward the input through the module
        :param x: the input to pass through the module
        :return: the processed input
        """
        self._activation_ = x.clone()
        to_mult = empty(x.shape).zero_().to(x.device).float()
        to_mult[x>=0]=1
        to_mult[x < 0]=self.alpha
        return to_mult.mul(x)
    
    def backward(self, *gradwrtoutput):
        """
        Return the gradient with respect to the input of the module
        :param gradwrtoutpt: the gradient with respect to the output of the current module
        :return: the gradient with respect to the input of the module
        """
        assert self._activation_ is not None, 'Forward pass must happen before backward pass'
        input_ = gradwrtoutput[0].clone()
        
        
        to_mult = self._activation_
        to_mult[to_mult >= 0] = 1
        to_mult[to_mult < 0] = self.alpha
        
        return to_mult.mul(input_)
    
class Sigmoid(Module):
    def __init__(self):
        """
        Build the sigmoid function
        """
        self._activation_ = None
        
    def forward(self, x):
        """
        Forward the input through the module
        :param x: the input to pass through the module
        :return: the processed input
        """
        self._activation_ = x.sigmoid().clone()
        return x.sigmoid()
        
    def backward(self, *gradwrtoutput):
        """
        Return the gradient with respect to the input of the module
        :param gradwrtoutpt: the gradient with respect to the output of the current module
        :return: the gradient with respect to the input of the module
        """
        input = gradwrtoutput[0]
        assert self._activation_ is not None, 'Forward pass must happen before backward pass'
        
        to_mult = self._activation_
        deri = to_mult*(1-to_mult)
        
        return deri.mul(input)