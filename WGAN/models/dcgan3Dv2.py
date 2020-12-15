import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data

      
class DCGAN_D(nn.Module):
    """ 
    discriminator component of WGAN
    """

    def __init__(self, ndf):
        super(DCGAN_D, self).__init__()    
        self.ndf = ndf
    
        
        ### convolution
        self.conv1 = torch.nn.Conv3d(1, ndf, kernel_size=(10,2,2), stride=(1,2,2), padding=(0,1,1), bias=False)
        ## layer-normalization
        self.bn1 = torch.nn.LayerNorm([39, 25, 25]) 
        ## convolution
        self.conv2 = torch.nn.Conv3d(ndf, ndf*2, kernel_size=(6,2,2), stride=(1,2,2), padding=0, bias=False)
        ## layer-normalization
        self.bn2 = torch.nn.LayerNorm([34, 12, 12])
        #convolution
        self.conv3 = torch.nn.Conv3d(ndf*2, ndf*4, kernel_size=(4,2,2), stride=(1,2,2), padding=0, bias=False)
        ## layer-normalization
        self.bn3 = torch.nn.LayerNorm([31, 6, 6])
        #convolution
        self.conv4 = torch.nn.Conv3d(ndf*4, ndf*2, kernel_size=(4,2,2), stride=1, padding=0, bias=False)
        #convolution
        self.bn4 = torch.nn.LayerNorm([28, 5, 5])
        ## layer-normalization
        self.conv5 = torch.nn.Conv3d(ndf*2, 1, kernel_size=(2,2,2), stride=(2,1,1), padding=0, bias=False)


        # Read-out layer : 1 * isize * isize input features, ndf output features 
        self.fc1 = torch.nn.Linear((14 * 4 * 4)+1, 100)
        self.fc2 = torch.nn.Linear(100, 50)
        self.fc3 = torch.nn.Linear(50, 1)
        

    def forward(self, x, energy):
        
        
        # N (Nlayers) x 48 x 48        
        
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = self.conv5(x)
        
        #Grand total --> size changes from (48, 48, 48) to (14, 4, 4)

        
        x = x.view(-1, 14 * 4 * 4)
        # Size changes from (14, 4, 4) to (1, 14 * 4 * 4) 
        #Recall that the -1 infers this dimension from the other given dimension

        energy = energy.view(-1,1)
        #print (x.shape, energy.shape)
        
        x = torch.cat((x, energy), 1)

        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
       
        
        # Read-out layer 
        output_wgan = self.fc3(x)
        
        #output_wgan = output_wgan.view(-1) ### flattens

        return output_wgan

class DCGAN_G(nn.Module):
    """ 
        generator component of WGAN
    """
    def __init__(self, ngf, nz):
        super(DCGAN_G, self).__init__()
        
        self.ngf = ngf
        self.nz = nz

        kernel = 4
        
        # input energy shape [batch x 1 x 1 x 1 ] going into convolutional
        self.conv1_1 = nn.ConvTranspose3d(1, ngf*4, kernel, 1, 0, bias=False)
        # state size [ ngf*4 x 4 x 4 x 4]
        
        # input noise shape [batch x nz x 1 x 1] going into convolutional
        self.conv1_100 = nn.ConvTranspose3d(nz, ngf*4, kernel, 1, 0, bias=False)
        # state size [ ngf*4 x 4 x 4 x 4]
        
        
        # outs from first convolutions concatenate state size [ ngf*8 x 4 x 4]
        # and going into main convolutional part of Generator
        self.main_conv = nn.Sequential(
            
            nn.ConvTranspose3d(ngf*8, ngf*4, kernel, 2, 1, bias=False),
            nn.BatchNorm3d(ngf*4),
            nn.ReLU(True),
            # state shape [ (ndf*4) x 8 x 8 ]

            nn.ConvTranspose3d(ngf*4, ngf*2, kernel, 2, 1, bias=False),
            nn.BatchNorm3d(ngf*2),
            nn.ReLU(True),
            # state shape [ (ndf*2) x 16 x 16 ]

            nn.ConvTranspose3d(ngf*2, ngf, kernel, 2, 1, bias=False),
            nn.BatchNorm3d(ngf),
            nn.ReLU(True),
            # state shape [ (ndf) x 32 x 32 ]

            nn.ConvTranspose3d(ngf, 10, 10, 1, 1, bias=False),
            nn.BatchNorm3d(10),
            nn.ReLU(True),
            # state shape [ 10 x 39 x 39 ]
           
            nn.ConvTranspose3d(10, 5, 8, 1, 1, bias=False),
            nn.BatchNorm3d(5),
            nn.ReLU(True),
            # state shape [ 5 x 44 x 44 ]
            
            nn.ConvTranspose3d(5, 1, 7, 1, 1, bias=False),
            nn.ReLU()
            
        )

    def forward(self, noise, energy):
        energy_trans = self.conv1_1(energy)
        noise_trans = self.conv1_100(noise)
        input = torch.cat((energy_trans, noise_trans), 1)
        x = self.main_conv(input)
        x = x.view(-1, 48, 48, 48)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('LayerNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm3d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)