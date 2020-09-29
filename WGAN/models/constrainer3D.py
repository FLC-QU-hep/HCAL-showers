import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data


class energyRegressor(nn.Module):
    """ 
    Energy regressor of WGAN. 
    """

    def __init__(self):
        super(energyRegressor, self).__init__()
        
        ## 3d conv layers
        self.conv1 = torch.nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn1 = torch.nn.LayerNorm([23,23,23])
        self.conv2 = torch.nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn2 = torch.nn.LayerNorm([11,11,11])
        self.conv3 = torch.nn.Conv3d(32, 16, kernel_size=2, stride=1, padding=0, bias=False)
 
       
        ## FC layers
        self.fc1 = torch.nn.Linear(16 * 10 * 10 * 10, 400)
        self.fc2 = torch.nn.Linear(400, 200)
        self.fc3 = torch.nn.Linear(200, 100)
        self.fc4 = torch.nn.Linear(100, 1)
        
    def forward(self, x):
        #input shape :  [48, 48, 48]
        ## reshape the input: expand one dim
        #x = x.unsqueeze(1)
        
        ## image [48, 48, 48]
        ### convolution adn batch normalisation
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = self.conv3(x)
      
        ## shape [10, 10, 10]
        
        ## flatten for FC
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3) * x.size(4))
        
        ## pass to FC layers
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.relu(self.fc4(x))
        return x