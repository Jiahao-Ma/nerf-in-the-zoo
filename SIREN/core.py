import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage import laplace, sobel

def paper_init(weight, is_first, omega=1):
    '''
        Initialize the weights of the neural layer
        # Parameters:
            weight: torch.Tensor
                the learnable 2D weight matrix
            is_first: boolean
                the bool value to determine if it is first layer of the hidden layer
            omega: int
                the scaling value
        # Returns:
            weight: torch.Tensor
                the learnable 2D weight matrix
    '''
    in_feats = weight.shape[1]
    with torch.no_grad():
        if is_first:
            bound = 1 / in_feats
        else:
            bound = np.sqrt( 6 / in_feats) / omega
        weight.uniform_(-bound, bound)
    

class SirenLayer(nn.Module):
    def __init__(self, in_feats, out_feats, bias, omega, is_first, custum_function_init=None):
        ''' 
            Linear layer followed by SIREAN function
            # Parameters:
                in_feats, out_feats: int
                    the channel number of input and the output
                bias: boolean 
                    the boolean value to determine whether it needs bias
                omega: int
                    the scaling value
                is_first: boolean
                    the bool value to determine if it is first layer of the hidden layer
                custum_function_init: `function` or boolean
                    init method
        '''
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_feats, out_feats, bias)
        if custum_function_init is None:
            paper_init(self.linear.weight, is_first, omega)
        else:
            custum_function_init(self.linear.weight)

    def forward(self, x):
        '''
            Forward. Linear layer followed by SIREAN function
            # Parameters:
                x: torch.Tensor
                    (n_samples, in_feats)
            # Returns:
                torch.Tensor with the shape of (n_samples, in_feats)
        '''
        return torch.sin( self.omega * self.linear(x) )

class SirenImage(nn.Module):
    def __init__(self, 
                hidden_feats,
                hidden_layers,
                bias,
                first_omega,
                hidden_omega,
                custum_function_init=None
                ):
        super().__init__()
        ''' The neural networks composed of Sirenlayer
        # Parameters:
            hidden_feats: int
                the input channel of hidden layers
            hidden_layers: int
                the number of hidden layers
            bias: boolean
                the bias of linear layer
            first_omega, hidden_omega: int
                the scaling value of first layer and hidden layer
            custum_function_init: boolean
                the initialize methods
        '''
        in_feats = 2
        out_feats = 1

        nets = list()
        nets.append(SirenLayer(in_feats, hidden_feats, bias, first_omega, True, custum_function_init))

        for _ in range(hidden_layers):
            nets.append(SirenLayer(hidden_feats, hidden_feats, bias, hidden_omega, False, custum_function_init))

        last_layer = nn.Linear(hidden_feats, out_feats, bias)
        if custum_function_init is None:
            paper_init(last_layer.weight, False, hidden_omega)
        else:
            custum_function_init(last_layer.weight)
        nets.append(last_layer)
        self.nets = nn.Sequential(*nets)
    
    def forward(self, x):
        '''
            Forward.
            # Parameters:
                x: torch.Tensor
                    (n_samples, n_feats)
            # Returns:
                torch.Tensor with the shape of (n_samples, n_feats)
        '''
        return self.nets(x)

def generate_coord_abs(n):
    ''' Generate the absolute coordinate 
        of image with the shape of (n, n)
        # Parameters:
            n: int, the size of the square image
        # Returns:
            coords: np.array
                (n*n, 2) the absolute coordinates of the square image
    '''
    cols, rows = np.meshgrid(range(n), range(n))
    coords = np.stack([cols.ravel(), rows.ravel()], axis=-1)
    return coords

class PixelDataset(Dataset):
    def __init__(self, img) -> None:
        super(PixelDataset, self).__init__()
        assert len(img.shape) == 2 and img.shape[0] == img.shape[1], \
            "the Dataset only support 2D square image."
        self.size = np.array(img.shape)
        '''
            Generate rgbs, Gradients (first-order derivatives), Laplacian (second-order derivatives)
        '''
        self.img = img
        self.coords_abs = generate_coord_abs(self.size[0])
        self.grads = np.stack([sobel(img, axis=0), sobel(img, axis=1)], axis=-1)
        self.grads_norm = np.linalg.norm(self.grads, axis=-1)
        self.laps = laplace(img)

    def __len__(self):
        # determine the number of the samples
        return self.size[0] * self.size[1]

    def __getitem__(self, index):
        # get the data for single coordinate
        coord_abs = self.coords_abs[index]
        r, c = coord_abs
        coord = 2 * ((coord_abs / self.size) - 0.5) # -1 ~ 1
        return { 'intensity' : self.img[r, c],
                 'grad'      : self.grads[r, c],
                 'laplace'   : self.laps[r, c],
                 'grad_norm' : self.grads_norm[r, c],
                 'coord'     : coord.astype(np.float32),
                 'coord_abs' : coord_abs
               }       

class GradientUtils:
    @staticmethod
    def gradient(target, coord):
        return torch.autograd.grad(target, coord, grad_outputs=torch.ones_like(target), create_graph=True)[0]

    @staticmethod
    def divergence(grads, coords):
        div = 0.0
        for i in range(coords.shape[1]):
            div += torch.autograd.grad(
                grads[..., i], coords, torch.ones_like(grads[..., i]), create_graph=True
            )[0][..., i : i + 1]
        return div

    @staticmethod
    def laplace(target, coords):
        grads = GradientUtils.gradient(target, coords)
        return GradientUtils.divergence(grads, coords)

