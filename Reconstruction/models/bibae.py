import numpy as np
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch import autograd

class BiBAE_F_3D_LayerNorm_SmallLatent(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, device, nc=1, ngf=8, z_rand=500, z_enc=12):
        super(BiBAE_F_3D_LayerNorm_SmallLatent, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        #self.z = z
        self.z_rand = z_rand
        self.z_enc = z_enc
        self.z_full = z_enc + z_rand
        self.args = args
        self.device = device

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.LayerNorm([24,24,24])
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.LayerNorm([12,12,12])
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.LayerNorm([6,6,6])
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=(3,3,3), stride=(1,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.LayerNorm([6,6,6])

     
        self.fc1 = nn.Linear(6*6*6*ngf*8+1, ngf*400, bias=True)
        self.fc2 = nn.Linear(ngf*400, int(self.z_full*1.5), bias=True)
        
        self.fc31 = nn.Linear(int(self.z_full*1.5), self.z_enc, bias=True)
        self.fc32 = nn.Linear(int(self.z_full*1.5), self.z_enc, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z_full+1, int(self.z_full*1.5), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z_full*1.5), ngf*400, bias=True)
        self.cond3 = torch.nn.Linear(ngf*400, 12*12*12*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.LayerNorm([24,24,24])
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf*2, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.LayerNorm([48,48,48])

        #self.deconv3 = torch.nn.ConvTranspose3d(ngf*4, ngf*8, kernel_size=(3,3,3), stride=(2,2,2), padding=(0,1,1), bias=False)
        
        self.conv0 = torch.nn.Conv3d(ngf*2, ngf, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco0 = torch.nn.LayerNorm([48,48,48])
        self.conv1 = torch.nn.Conv3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco1 = torch.nn.LayerNorm([48,48,48])
        self.conv2 = torch.nn.Conv3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco2 = torch.nn.LayerNorm([48,48,48])
        self.conv3 = torch.nn.Conv3d(ngf*4, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.bnco3 = torch.nn.LayerNorm([48,48,48])
        self.conv4 = torch.nn.Conv3d(ngf*2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
    
        #self.dr03 = nn.Dropout(p=0.3, inplace=False)
        #self.dr05 = nn.Dropout(p=0.5, inplace=False)

    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,48,48,48))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        return torch.cat((self.fc31(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1), torch.cat((self.fc32(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        #print(std)
        #print(mu)
        return mu + eps*std

    def decode(self, z):
        ### need to do generated 30 layers, hence the loop!
        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond1(z), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond2(x), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond3(x), 0.2, inplace=True)

        ## change size for deconv2d network. Image is 10x10
        x = x.view(-1,self.ngf,12,12,12)        

        ## apply series of deconv2d and batch-norm
        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 24, 24, 24])), 0.2, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 48, 48, 48])), 0.2, inplace=True) #
        #x = F.leaky_relu(self.deconv3(x, output_size=[x.size(0), 1, 15, 120, 120]), 0.2, inplace=True) #

        ##Image is 120x120
        x = F.leaky_relu(self.bnco0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco3(self.conv3(x)), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        return x 
        
        
    def forward(self, x, E_true, z=None, mode='full'):
        #print(x.size())
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z 
        elif mode == 'decode':
            return self.decode(torch.cat((z,E_true), 1)) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), mu, logvar, z
        

        
        
class Discriminator_F_Conv_DIRENR_Diff_v3LinOut(nn.Module):
    def __init__(self, isize=48, nc=2, ndf=64):
        super(Discriminator_F_Conv_DIRENR_Diff_v3LinOut, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc

        
        self.conv1b = torch.nn.Conv3d(1, ndf, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1b = torch.nn.LayerNorm([24,24,24])
        self.conv2b = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2b = torch.nn.LayerNorm([12,12,12])
        self.conv3b = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False)


        self.conv1c = torch.nn.Conv3d(1, ndf, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1c = torch.nn.LayerNorm([24,24,24])
        self.conv2c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2c = torch.nn.LayerNorm([12,12,12])
        self.conv3c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False)

 
        #self.fc = torch.nn.Linear(5 * 4 * 4, 1)
        self.fc1a = torch.nn.Linear(48*48*48, int(ndf/2)) 
        self.fc1b = torch.nn.Linear(ndf * 6 * 6 * 6, int(ndf/2))
        self.fc1c = torch.nn.Linear(ndf * 6 * 6 * 6, int(ndf/2))
        self.fc1e = torch.nn.Linear(1, int(ndf/2))
        self.fc2 = torch.nn.Linear(int(ndf/2)*4, ndf*2)
        self.fc3 = torch.nn.Linear(ndf*2, ndf*2)
        self.fc4 = torch.nn.Linear(ndf*2, ndf*2)
        self.fc5 = torch.nn.Linear(ndf*2, 1)


    def forward(self, img, E_true):
        imga = img[:,0:1,:,:,:] #data.view(-1,1,30,30)
        imgb = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)

        img = imgb - imga
        
        xb = F.leaky_relu(self.bn1b(self.conv1b(imgb)), 0.2)
        xb = F.leaky_relu(self.bn2b(self.conv2b(xb)), 0.2)
        xb = F.leaky_relu(self.conv3b(xb), 0.2)
        xb = xb.view(-1, self.ndf * 6 * 6 * 6)
        xb = F.leaky_relu(self.fc1b(xb), 0.2)        
        
        imgb_relu = F.relu(imgb)
        
        xc = F.leaky_relu(self.bn1c(self.conv1c(torch.log(imgb_relu+1.0))), 0.2)
        xc = F.leaky_relu(self.bn2c(self.conv2c(xc)), 0.2)
        xc = F.leaky_relu(self.conv3c(xc), 0.2)
        xc = xc.view(-1, self.ndf * 6 * 6 * 6)
        xc = F.leaky_relu(self.fc1c(xc), 0.2)
        
        xb = torch.cat((xb, xc, F.leaky_relu(self.fc1a(img.view(-1, 48*48*48))) , F.leaky_relu(self.fc1e(E_true), 0.2)), 1)

        xb = F.leaky_relu(self.fc2(xb), 0.2)
        xb = F.leaky_relu(self.fc3(xb), 0.2)
        xb = F.leaky_relu(self.fc4(xb), 0.2)
        xb = self.fc5(xb)

        return xb.view(-1) ### flattens
    
    
    
class PostProcess_Size1Conv_EcondV2(nn.Module):
    def __init__(self, isize=48, nc=2, ndf=64, bias=False, out_funct='relu'):
        super(PostProcess_Size1Conv_EcondV2, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc
        self.bais = bias
        self.out_funct = out_funct
        
        self.fcec1 = torch.nn.Linear(2, int(ndf/2), bias=True)
        self.fcec2 = torch.nn.Linear(int(ndf/2), int(ndf/2), bias=True)
        self.fcec3 = torch.nn.Linear(int(ndf/2), int(ndf/2), bias=True)
        
        self.conv1 = torch.nn.Conv3d(1, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco1 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])
        
        self.conv2 = torch.nn.Conv3d(ndf+int(ndf/2), ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco2 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])

        self.conv3 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco3 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])

        self.conv4 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco4 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])

        self.conv5 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)

        self.conv6 = torch.nn.Conv3d(ndf, 1, kernel_size=1, stride=1, padding=0, bias=False)
 

    def forward(self, img, E_True=0):
        img = img.view(-1, 1, self.isize, self.isize, self.isize)
                
        econd = torch.cat((torch.sum(img.view(-1, self.isize*self.isize*self.isize), 1).view(-1, 1), E_True), 1)

        econd = F.leaky_relu(self.fcec1(econd), 0.01)
        econd = F.leaky_relu(self.fcec2(econd), 0.01)
        econd = F.leaky_relu(self.fcec3(econd), 0.01)
        
        econd = econd.view(-1, int(self.ndf/2), 1, 1, 1)
        econd = econd.expand(-1, -1, self.isize, self.isize, self.isize)        
        
        img = F.leaky_relu(self.bnco1(self.conv1(img)), 0.01)
        img = torch.cat((img, econd), 1)
        img = F.leaky_relu(self.bnco2(self.conv2(img)), 0.01)
        img = F.leaky_relu(self.bnco3(self.conv3(img)), 0.01)
        img = F.leaky_relu(self.bnco4(self.conv4(img)), 0.01)
        img = F.leaky_relu(self.conv5(img), 0.01) 
        img = self.conv6(img)

        if self.out_funct == 'relu':
            img = F.relu(img)
        elif self.out_funct == 'leaky_relu':
            img = F.leaky_relu(img, 0.01) 
              
        return img.view(-1, 1, self.isize, self.isize, self.isize)


            
            
    
    
    
class Latent_Critic(nn.Module):
    def __init__(self, ):
        super(Latent_Critic, self).__init__()
        self.linear1 = nn.Linear(1, 50)
        self.linear2 = nn.Linear(50, 100)        
        self.linear3 = nn.Linear(100, 50)
        self.linear4 = nn.Linear(50, 1)

    def forward(self, x):      
        x = F.leaky_relu(self.linear1(x.view(-1,1)), inplace=True)
        x = F.leaky_relu(self.linear2(x), inplace=True)
        x = F.leaky_relu(self.linear3(x), inplace=True)
        return self.linear4(x)
     

        
        
        
        
        
        
        
        
class BiBAE_F_3D_LayerNorm_SmallLatent_LogScale(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, device, nc=1, ngf=8, z_rand=500, z_enc=12):
        super(BiBAE_F_3D_LayerNorm_SmallLatent_LogScale, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        #self.z = z
        self.z_rand = z_rand
        self.z_enc = z_enc
        self.z_full = z_enc + z_rand
        self.args = args
        self.device = device

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.LayerNorm([24,24,24])
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.LayerNorm([12,12,12])
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.LayerNorm([6,6,6])
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=(3,3,3), stride=(1,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.LayerNorm([6,6,6])

     
        self.fc1 = nn.Linear(6*6*6*ngf*8+1, ngf*400, bias=True)
        self.fc2 = nn.Linear(ngf*400, int(self.z_full*1.5), bias=True)
        
        self.fc31 = nn.Linear(int(self.z_full*1.5), self.z_enc, bias=True)
        self.fc32 = nn.Linear(int(self.z_full*1.5), self.z_enc, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z_full+1, int(self.z_full*1.5), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z_full*1.5), ngf*400, bias=True)
        self.cond3 = torch.nn.Linear(ngf*400, 12*12*12*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.LayerNorm([24,24,24])
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf*2, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.LayerNorm([48,48,48])

        #self.deconv3 = torch.nn.ConvTranspose3d(ngf*4, ngf*8, kernel_size=(3,3,3), stride=(2,2,2), padding=(0,1,1), bias=False)
        
        self.conv0 = torch.nn.Conv3d(ngf*2, ngf, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco0 = torch.nn.LayerNorm([48,48,48])
        self.conv1 = torch.nn.Conv3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco1 = torch.nn.LayerNorm([48,48,48])
        self.conv2 = torch.nn.Conv3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco2 = torch.nn.LayerNorm([48,48,48])
        self.conv3 = torch.nn.Conv3d(ngf*4, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.bnco3 = torch.nn.LayerNorm([48,48,48])
        self.conv4 = torch.nn.Conv3d(ngf*2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
    
        #self.dr03 = nn.Dropout(p=0.3, inplace=False)
        #self.dr05 = nn.Dropout(p=0.5, inplace=False)

    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,48,48,48))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        return torch.cat((self.fc31(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1), torch.cat((self.fc32(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        #print(std)
        #print(mu)
        return mu + eps*std

    def decode(self, z):
        ### need to do generated 30 layers, hence the loop!
        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond1(z), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond2(x), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond3(x), 0.2, inplace=True)

        ## change size for deconv2d network. Image is 10x10
        x = x.view(-1,self.ngf,12,12,12)        

        ## apply series of deconv2d and batch-norm
        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 24, 24, 24])), 0.2, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 48, 48, 48])), 0.2, inplace=True) #
        #x = F.leaky_relu(self.deconv3(x, output_size=[x.size(0), 1, 15, 120, 120]), 0.2, inplace=True) #

        ##Image is 120x120
        x = F.leaky_relu(self.bnco0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco3(self.conv3(x)), 0.2, inplace=True)
        x = self.conv4(x)
        return x 
        
        
    def forward(self, x, E_true, z=None, mode='full'):
        #print(x.size())
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z 
        elif mode == 'decode':
            return self.decode(torch.cat((z,E_true), 1)) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), mu, logvar, z
        

        
        
class Discriminator_F_Conv_DIRENR_Diff_v3_LogScale(nn.Module):
    def __init__(self, isize=48, nc=2, ndf=64):
        super(Discriminator_F_Conv_DIRENR_Diff_v3_LogScale, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc

        
        self.conv1b = torch.nn.Conv3d(1, ndf, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1b = torch.nn.LayerNorm([24,24,24])
        self.conv2b = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2b = torch.nn.LayerNorm([12,12,12])
        self.conv3b = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False)


        self.conv1c = torch.nn.Conv3d(1, ndf, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1c = torch.nn.LayerNorm([24,24,24])
        self.conv2c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2c = torch.nn.LayerNorm([12,12,12])
        self.conv3c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False)

 
        #self.fc = torch.nn.Linear(5 * 4 * 4, 1)
        self.fc1a = torch.nn.Linear(48*48*48, int(ndf/2)) 
        self.fc1b = torch.nn.Linear(ndf * 6 * 6 * 6, int(ndf/2))
        self.fc1c = torch.nn.Linear(ndf * 6 * 6 * 6, int(ndf/2))
        self.fc1e = torch.nn.Linear(1, int(ndf/2))
        self.fc2 = torch.nn.Linear(int(ndf/2)*4, ndf*2)
        self.fc3 = torch.nn.Linear(ndf*2, ndf*2)
        self.fc4 = torch.nn.Linear(ndf*2, ndf*2)
        self.fc5 = torch.nn.Linear(ndf*2, 1)


    def forward(self, img, E_true):
        imga = img[:,0:1,:,:,:] #data.view(-1,1,30,30)
        imgb = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)

        img = imgb - imga
        
        xb = F.leaky_relu(self.bn1b(self.conv1b(imgb)), 0.2)
        xb = F.leaky_relu(self.bn2b(self.conv2b(xb)), 0.2)
        xb = F.leaky_relu(self.conv3b(xb), 0.2)
        xb = xb.view(-1, self.ndf * 6 * 6 * 6)
        xb = F.leaky_relu(self.fc1b(xb), 0.2)        
        
        #imgb_relu = F.relu(imgb)
        
        xc = F.leaky_relu(self.bn1c(self.conv1c(torch.exp(imgb)-0.00001)), 0.2)
        xc = F.leaky_relu(self.bn2c(self.conv2c(xc)), 0.2)
        xc = F.leaky_relu(self.conv3c(xc), 0.2)
        xc = xc.view(-1, self.ndf * 6 * 6 * 6)
        xc = F.leaky_relu(self.fc1c(xc), 0.2)
        
        xb = torch.cat((xb, xc, F.leaky_relu(self.fc1a(img.view(-1, 48*48*48))) , F.leaky_relu(self.fc1e(E_true), 0.2)), 1)

        xb = F.leaky_relu(self.fc2(xb), 0.2)
        xb = F.leaky_relu(self.fc3(xb), 0.2)
        xb = F.leaky_relu(self.fc4(xb), 0.2)
        xb = self.fc5(xb)

        return xb.view(-1) ### flattens
    
    
    
    
    
class BiBAE_F_3D_LayerNorm_SmallLatent_LogScaleThresh_V2(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, device, nc=1, ngf=8, z_rand=500, z_enc=12):
        super(BiBAE_F_3D_LayerNorm_SmallLatent_LogScaleThresh_V2, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        #self.z = z
        self.z_rand = z_rand
        self.z_enc = z_enc
        self.z_full = z_enc + z_rand
        self.args = args
        self.device = device

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.LayerNorm([24,24,24])
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.LayerNorm([12,12,12])
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.LayerNorm([6,6,6])
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=(3,3,3), stride=(1,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.LayerNorm([6,6,6])

     
        self.fc1 = nn.Linear(6*6*6*ngf*8+1, ngf*400, bias=True)
        self.fc2 = nn.Linear(ngf*400, int(self.z_full*1.5), bias=True)
        
        self.fc31 = nn.Linear(int(self.z_full*1.5), self.z_enc, bias=True)
        self.fc32 = nn.Linear(int(self.z_full*1.5), self.z_enc, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z_full+1, int(self.z_full*1.5), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z_full*1.5), ngf*400, bias=True)
        self.cond3 = torch.nn.Linear(ngf*400, 12*12*12*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.LayerNorm([24,24,24])
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf*2, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.LayerNorm([48,48,48])

        #self.deconv3 = torch.nn.ConvTranspose3d(ngf*4, ngf*8, kernel_size=(3,3,3), stride=(2,2,2), padding=(0,1,1), bias=False)
        
        self.conv0 = torch.nn.Conv3d(ngf*2, ngf, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco0 = torch.nn.LayerNorm([48,48,48])
        self.conv1 = torch.nn.Conv3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco1 = torch.nn.LayerNorm([48,48,48])
        self.conv2 = torch.nn.Conv3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco2 = torch.nn.LayerNorm([48,48,48])
        self.conv3 = torch.nn.Conv3d(ngf*4, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.bnco3 = torch.nn.LayerNorm([48,48,48])
        self.conv4 = torch.nn.Conv3d(ngf*2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
    
        #self.dr03 = nn.Dropout(p=0.3, inplace=False)
        #self.dr05 = nn.Dropout(p=0.5, inplace=False)

    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,48,48,48))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        return torch.cat((self.fc31(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1), torch.cat((self.fc32(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        #print(std)
        #print(mu)
        return mu + eps*std

    def decode(self, z):
        ### need to do generated 30 layers, hence the loop!
        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond1(z), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond2(x), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond3(x), 0.2, inplace=True)

        ## change size for deconv2d network. Image is 10x10
        x = x.view(-1,self.ngf,12,12,12)        

        ## apply series of deconv2d and batch-norm
        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 24, 24, 24])), 0.2, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 48, 48, 48])), 0.2, inplace=True) #
        #x = F.leaky_relu(self.deconv3(x, output_size=[x.size(0), 1, 15, 120, 120]), 0.2, inplace=True) #

        ##Image is 120x120
        x = F.leaky_relu(self.bnco0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco3(self.conv3(x)), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        return x 
        
        
    def forward(self, x, E_true, z=None, mode='full'):
        #print(x.size())
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z 
        elif mode == 'decode':
            return self.decode(torch.cat((z,E_true), 1)) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), mu, logvar, z
        

    
    
    
class Discriminator_F_Conv_DIRENR_Diff_v3_LogScaleThresh_V2(nn.Module):
    def __init__(self, isize=48, nc=2, ndf=64):
        super(Discriminator_F_Conv_DIRENR_Diff_v3_LogScaleThresh_V2, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc

        
        self.conv1b = torch.nn.Conv3d(1, ndf, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1b = torch.nn.LayerNorm([24,24,24])
        self.conv2b = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2b = torch.nn.LayerNorm([12,12,12])
        self.conv3b = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False)


        self.conv1c = torch.nn.Conv3d(1, ndf, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1c = torch.nn.LayerNorm([24,24,24])
        self.conv2c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2c = torch.nn.LayerNorm([12,12,12])
        self.conv3c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False)

 
        #self.fc = torch.nn.Linear(5 * 4 * 4, 1)
        self.fc1a = torch.nn.Linear(48*48*48, int(ndf/2)) 
        self.fc1b = torch.nn.Linear(ndf * 6 * 6 * 6, int(ndf/2))
        self.fc1c = torch.nn.Linear(ndf * 6 * 6 * 6, int(ndf/2))
        self.fc1e = torch.nn.Linear(1, int(ndf/2))
        self.fc2 = torch.nn.Linear(int(ndf/2)*4, ndf*2)
        self.fc3 = torch.nn.Linear(ndf*2, ndf*2)
        self.fc4 = torch.nn.Linear(ndf*2, ndf*2)
        self.fc5 = torch.nn.Linear(ndf*2, 1)


    def forward(self, img, E_true):
        imga = img[:,0:1,:,:,:] #data.view(-1,1,30,30)
        imgb = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)

        img = imgb - imga
        
        xb = F.leaky_relu(self.bn1b(self.conv1b(imgb)), 0.2)
        xb = F.leaky_relu(self.bn2b(self.conv2b(xb)), 0.2)
        xb = F.leaky_relu(self.conv3b(xb), 0.2)
        xb = xb.view(-1, self.ndf * 6 * 6 * 6)
        xb = F.leaky_relu(self.fc1b(xb), 0.2)        
        
        #imgb_relu = F.relu(imgb)
        
        xc = F.leaky_relu(self.bn1c(self.conv1c(torch.exp(imgb-5.0)-np.exp(-5.0))), 0.2)
        xc = F.leaky_relu(self.bn2c(self.conv2c(xc)), 0.2)
        xc = F.leaky_relu(self.conv3c(xc), 0.2)
        xc = xc.view(-1, self.ndf * 6 * 6 * 6)
        xc = F.leaky_relu(self.fc1c(xc), 0.2)
        
        xb = torch.cat((xb, xc, F.leaky_relu(self.fc1a(img.view(-1, 48*48*48))) , F.leaky_relu(self.fc1e(E_true), 0.2)), 1)

        xb = F.leaky_relu(self.fc2(xb), 0.2)
        xb = F.leaky_relu(self.fc3(xb), 0.2)
        xb = F.leaky_relu(self.fc4(xb), 0.2)
        xb = self.fc5(xb)

        return xb.view(-1) ### flattens

    
    
    
    
    
    
    
    
    
    
    
    
class BiBAE_F_3D_BatchStat_LogScale(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, device, nc=1, ngf=8, z_rand=500, z_enc=12):
        super(BiBAE_F_3D_BatchStat_LogScale, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        self.z_rand = z_rand
        self.z_enc = z_enc
        self.z_full = z_enc + z_rand
        self.args = args
        self.device = device

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(2,2,2), stride=(2,2,2),
                               padding=(0,0,0), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.LayerNorm([24,24,24])
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(2,2,2), stride=(2,2,2),
                               padding=(0,0,0), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.LayerNorm([12,12,12])
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.LayerNorm([6,6,6])
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*4, kernel_size=(4,4,4), stride=(1,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.LayerNorm([5,5,5])

     
        self.fc1 = nn.Linear(5*5*5*ngf*4+1, ngf*100, bias=True)
        self.fc2 = nn.Linear(ngf*100, int(self.z_full), bias=True)
        
        self.fc31 = nn.Linear(int(self.z_full), self.z_enc, bias=True)
        self.fc32 = nn.Linear(int(self.z_full), self.z_enc, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z_full+1, int(self.z_full), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z_full), ngf*100, bias=True)
        self.cond3 = torch.nn.Linear(ngf*100, 12*12*12*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.LayerNorm([24,24,24])
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.LayerNorm([48,48,48])
        
        self.conv0 = torch.nn.Conv3d(ngf, ngf, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco0 = torch.nn.LayerNorm([48,48,48])
        self.conv1 = torch.nn.Conv3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco1 = torch.nn.LayerNorm([48,48,48])
        self.conv2 = torch.nn.Conv3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco2 = torch.nn.LayerNorm([48,48,48])
        self.conv3 = torch.nn.Conv3d(ngf*4, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.bnco3 = torch.nn.LayerNorm([48,48,48])
        self.conv4 = torch.nn.Conv3d(ngf*2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)


    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,48,48,48))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        return torch.cat((self.fc31(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1), torch.cat((self.fc32(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)

        x = x.view(-1,self.ngf,12,12,12)        

        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 24, 24, 24])), 0.2, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 48, 48, 48])), 0.2, inplace=True) #

        x = F.leaky_relu(self.bnco0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco3(self.conv3(x)), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        return x 
        
        
    def forward(self, x, E_true, z=None, mode='full'):
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z 
        elif mode == 'decode':
            return self.decode(torch.cat((z,E_true), 1)) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), mu, logvar, z
            
        

    
    
    
class BiBAE_F_3D_BatchStat_LogScale_LinNoise(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, device, nc=1, ngf=8, z_rand=500, z_enc=12):
        super(BiBAE_F_3D_BatchStat_LogScale_LinNoise, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        self.z_rand = z_rand
        self.z_enc = z_enc
        self.z_full = z_enc + z_rand
        self.args = args
        self.device = device

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(2,2,2), stride=(2,2,2),
                               padding=(0,0,0), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.LayerNorm([24,24,24])
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(2,2,2), stride=(2,2,2),
                               padding=(0,0,0), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.LayerNorm([12,12,12])
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.LayerNorm([6,6,6])
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*4, kernel_size=(4,4,4), stride=(1,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.LayerNorm([5,5,5])

     
        self.fc1 = nn.Linear(5*5*5*ngf*4+1, ngf*100, bias=True)
        self.fc2 = nn.Linear(ngf*100, int(self.z_full), bias=True)
        
        self.fc31 = nn.Linear(int(self.z_full), self.z_enc, bias=True)
        self.fc32 = nn.Linear(int(self.z_full), self.z_enc, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z_full+1, int(self.z_full), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z_full), ngf*100, bias=True)
        self.cond3 = torch.nn.Linear(ngf*100, 12*12*12*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.LayerNorm([24,24,24])
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.LayerNorm([48,48,48])
        
        self.conv0 = torch.nn.Conv3d(ngf, ngf, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco0 = torch.nn.LayerNorm([48,48,48])
        self.conv1 = torch.nn.Conv3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco1 = torch.nn.LayerNorm([48,48,48])
        self.conv2 = torch.nn.Conv3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco2 = torch.nn.LayerNorm([48,48,48])
        self.conv3 = torch.nn.Conv3d(ngf*4, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.bnco3 = torch.nn.LayerNorm([48,48,48])
        self.conv4 = torch.nn.Conv3d(ngf*2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)


    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,48,48,48))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        return torch.cat((self.fc31(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1), torch.cat((self.fc32(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, E_true):
        x1 = z[:,:self.z_enc]
        x2 = (torch.rand(z.size(0), int(self.z_rand), device=E_true.get_device())-0.5)
        z = torch.cat((x1, x2), 1)
        
        z = torch.cat((z,E_true), 1)

        
        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)

        x = x.view(-1,self.ngf,12,12,12)        

        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 24, 24, 24])), 0.2, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 48, 48, 48])), 0.2, inplace=True) #

        x = F.leaky_relu(self.bnco0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco3(self.conv3(x)), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        return x 
        
        
    def forward(self, x, E_true, z=None, mode='full'):
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z 
        elif mode == 'decode':
            return self.decode(z,E_true) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(z,E_true), mu, logvar, z
            
        

    
    
    
class Discriminator_F_BatchStat_LogScale(nn.Module):
    def __init__(self, isize=48, nc=2, ndf=64):
        super(Discriminator_F_BatchStat_LogScale, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc
        self.size_embed = 16
        self.conv1_bias = False

        self.Embed1a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed1b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed1c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed2a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed2b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed2c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed3a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed3b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed3c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed4a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed4b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed4c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        
        self.embed_lin = torch.nn.Linear(self.size_embed*4, 64) 
        self.embed_log = torch.nn.Linear(self.size_embed*4, 64) 
        self.diff_lin = torch.nn.Linear(48*48*48, 64) 
        self.diff_log = torch.nn.Linear(48*48*48, 64) 
        
        
        self.conv1a = torch.nn.Conv3d(1, ndf, kernel_size=2, stride=2, padding=0, bias=False)
        self.ln1a = torch.nn.LayerNorm([24,24,24])
        self.conv1b = torch.nn.Conv3d(ndf, ndf, kernel_size=2, stride=2, padding=0, bias=False)
        self.ln1b = torch.nn.LayerNorm([12,12,12])
        self.conv1c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.conv2a = torch.nn.Conv3d(1, ndf, kernel_size=2, stride=2, padding=0, bias=False)
        self.ln2a = torch.nn.LayerNorm([24,24,24])
        self.conv2b = torch.nn.Conv3d(ndf, ndf, kernel_size=2, stride=2, padding=0, bias=False)
        self.ln2b = torch.nn.LayerNorm([12,12,12])
        self.conv2c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False)

        self.conv_lin = torch.nn.Linear(6*6*6*ndf, 64) 
        self.conv_log = torch.nn.Linear(6*6*6*ndf, 64) 
        
        self.econd_lin = torch.nn.Linear(1, 64) 

        self.fc1 = torch.nn.Linear(64*7, 256)
        self.fc2 = torch.nn.Linear(256,  256)
        self.fc3 = torch.nn.Linear(256, 1)


    def forward(self, img, E_true):
        batch_size = img.size(0)
        
        img_log_a = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)
        img_log_b = img[:,0:1,:,:,:] #data.view(-1,1,30,30)

        #print('img_log_a', torch.sum(img_log_a))
        #print('img_log_b', torch.sum(img_log_b))

        img_a = torch.exp(img_log_a-5.0)-np.exp(-5.0)
        img_b = torch.exp(img_log_b-5.0)-np.exp(-5.0)
        
        
        img_a_hit_std = torch.std(img_a.view(batch_size, -1), 1).view(1, batch_size, 1).repeat(batch_size, 1, 1)*0.01
        img_a_hit_sum = torch.sum(img_a.view(batch_size, -1), 1).view(1, batch_size, 1).repeat(batch_size, 1, 1)*0.001
        
        img_a_hit_std = torch.transpose(img_a_hit_std, 1, 2)
        img_a_hit_sum = torch.transpose(img_a_hit_sum, 1, 2)

        
        img_a_hit_std = F.leaky_relu(self.Embed1a(img_a_hit_std))
        img_a_hit_std = F.leaky_relu(self.Embed1b(img_a_hit_std))
        img_a_hit_std = F.leaky_relu(self.Embed1c(img_a_hit_std))
        
        img_a_hit_sum = F.leaky_relu(self.Embed2a(img_a_hit_sum))
        img_a_hit_sum = F.leaky_relu(self.Embed2b(img_a_hit_sum))
        img_a_hit_sum = F.leaky_relu(self.Embed2c(img_a_hit_sum))

        img_a_hit_std_std  =  torch.std(img_a_hit_std, 2)
        img_a_hit_std_mean = torch.mean(img_a_hit_std, 2)

        img_a_hit_sum_std  =  torch.std(img_a_hit_sum, 2)
        img_a_hit_sum_mean = torch.mean(img_a_hit_sum, 2)
        
        xa1 = torch.cat((img_a_hit_std_std, img_a_hit_std_mean, img_a_hit_sum_std, img_a_hit_sum_mean), 1)
        xa1 = F.leaky_relu(self.embed_lin(xa1), 0.2)
        
        
        img_log_a_hit_std = torch.std(img_log_a.view(batch_size, -1), 1).view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_log_a_hit_sum = torch.sum(img_log_a.view(batch_size, -1), 1).view(1, batch_size, 1).repeat(batch_size, 1, 1)*0.1
        
        img_log_a_hit_std = torch.transpose(img_log_a_hit_std, 1, 2)
        img_log_a_hit_sum = torch.transpose(img_log_a_hit_sum, 1, 2)
        
        img_log_a_hit_std = F.leaky_relu(self.Embed1a(img_log_a_hit_std))
        img_log_a_hit_std = F.leaky_relu(self.Embed1b(img_log_a_hit_std))
        img_log_a_hit_std = F.leaky_relu(self.Embed1c(img_log_a_hit_std))
        
        img_log_a_hit_sum = F.leaky_relu(self.Embed2a(img_log_a_hit_sum))
        img_log_a_hit_sum = F.leaky_relu(self.Embed2b(img_log_a_hit_sum))
        img_log_a_hit_sum = F.leaky_relu(self.Embed2c(img_log_a_hit_sum))

        img_log_a_hit_std_std  =  torch.std(img_log_a_hit_std, 2)
        img_log_a_hit_std_mean = torch.mean(img_log_a_hit_std, 2)

        img_log_a_hit_sum_std  =  torch.std(img_log_a_hit_sum, 2)
        img_log_a_hit_sum_mean = torch.mean(img_log_a_hit_sum, 2)        
        
        xa2 = torch.cat((img_log_a_hit_std_std, img_log_a_hit_std_mean, img_log_a_hit_sum_std, img_log_a_hit_sum_mean), 1)
        xa2 = F.leaky_relu(self.embed_lin(xa2), 0.2)
        
               
        xa3 = F.leaky_relu(self.diff_lin((img_a-img_b).view(batch_size, -1)), 0.2)
        
        
        xa4 = F.leaky_relu(self.diff_log((img_log_a-img_log_b).view(batch_size, -1)), 0.2)
 

        xa5 = F.leaky_relu(self.ln1a(self.conv1a(img_a*0.001)), 0.2)
        xa5 = F.leaky_relu(self.ln1b(self.conv1b(xa5)), 0.2)
        xa5 = F.leaky_relu(self.conv1c(xa5), 0.2)
        xa5 = xa5.view(-1, self.ndf*6*6*6)
        xa5 = F.leaky_relu(self.conv_lin(xa5), 0.2)
        

        xa6 = F.leaky_relu(self.ln2a(self.conv2a(img_log_a)), 0.2)
        xa6 = F.leaky_relu(self.ln2b(self.conv2b(xa6)), 0.2)
        xa6 = F.leaky_relu(self.conv2c(xa6), 0.2)
        xa6 = xa6.view(-1, self.ndf*6*6*6)
        xa6 = F.leaky_relu(self.conv_log(xa6), 0.2)        
        
        xa7 = F.leaky_relu(self.econd_lin(E_true), 0.2)      
        
        xa = torch.cat((xa1, xa2, xa3, xa4, xa5, xa6, xa7), 1)
        
        #print('xa1', torch.sum(xa1))
        #print('xa2', torch.sum(xa2))
        #print('xa3', torch.sum(xa3))
        #print('xa4', torch.sum(xa4))
        #print('xa5', torch.sum(xa5))
        #print('xa6', torch.sum(xa6))
        #print('xa7', torch.sum(xa7))
        
        xa = F.leaky_relu(self.fc1(xa), 0.2)
        xa = F.leaky_relu(self.fc2(xa), 0.2)
        xa = self.fc3(xa)

        #print('xa', torch.sum(xa))
        
        return xa ### flattens
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
class BiBAE_F_3D_BatchStat(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, device, nc=1, ngf=8, z_rand=500, z_enc=12):
        super(BiBAE_F_3D_BatchStat, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        self.z_rand = z_rand
        self.z_enc = z_enc
        self.z_full = z_enc + z_rand
        self.args = args
        self.device = device

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(2,2,2), stride=(2,2,2),
                               padding=(0,0,0), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.LayerNorm([24,24,24])
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(2,2,2), stride=(2,2,2),
                               padding=(0,0,0), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.LayerNorm([12,12,12])
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.LayerNorm([6,6,6])
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*4, kernel_size=(4,4,4), stride=(1,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.LayerNorm([5,5,5])

     
        self.fc1 = nn.Linear(5*5*5*ngf*4+1, ngf*100, bias=True)
        self.fc2 = nn.Linear(ngf*100, int(self.z_full), bias=True)
        
        self.fc31 = nn.Linear(int(self.z_full), self.z_enc, bias=True)
        self.fc32 = nn.Linear(int(self.z_full), self.z_enc, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z_full+1, int(self.z_full), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z_full), ngf*100, bias=True)
        self.cond3 = torch.nn.Linear(ngf*100, 12*12*12*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.LayerNorm([24,24,24])
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.LayerNorm([48,48,48])
        
        self.conv0 = torch.nn.Conv3d(ngf, ngf, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco0 = torch.nn.LayerNorm([48,48,48])
        self.conv1 = torch.nn.Conv3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco1 = torch.nn.LayerNorm([48,48,48])
        self.conv2 = torch.nn.Conv3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco2 = torch.nn.LayerNorm([48,48,48])
        self.conv3 = torch.nn.Conv3d(ngf*4, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.bnco3 = torch.nn.LayerNorm([48,48,48])
        self.conv4 = torch.nn.Conv3d(ngf*2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)


    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,48,48,48))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        return torch.cat((self.fc31(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1), torch.cat((self.fc32(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)

        x = x.view(-1,self.ngf,12,12,12)        

        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 24, 24, 24])), 0.2, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 48, 48, 48])), 0.2, inplace=True) #

        x = F.leaky_relu(self.bnco0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco3(self.conv3(x)), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        return x 
        
        
    def forward(self, x, E_true, z=None, mode='full'):
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z 
        elif mode == 'decode':
            return self.decode(torch.cat((z,E_true), 1)) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), mu, logvar, z
            
        

    
    
    
class Discriminator_F_BatchStat(nn.Module):
    def __init__(self, isize=48, nc=2, ndf=64):
        super(Discriminator_F_BatchStat, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc
        self.size_embed = 16

        self.Embed1a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = True)
        self.Embed1b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = True)
        self.Embed1c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = True)

        self.Embed2a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = True)
        self.Embed2b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = True)
        self.Embed2c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = True)

        self.Embed3a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = True)
        self.Embed3b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = True)
        self.Embed3c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = True)

        self.Embed4a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = True)
        self.Embed4b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = True)
        self.Embed4c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = True)
        
        self.embed_lin = torch.nn.Linear(self.size_embed*4, 64) 
        self.embed_log = torch.nn.Linear(self.size_embed*4, 64) 
        self.diff_lin = torch.nn.Linear(48*48*48, 64) 
        self.diff_log = torch.nn.Linear(48*48*48, 64) 
        
        
        self.conv1a = torch.nn.Conv3d(1, ndf, kernel_size=2, stride=2, padding=0, bias=False)
        self.ln1a = torch.nn.LayerNorm([24,24,24])
        self.conv1b = torch.nn.Conv3d(ndf, ndf, kernel_size=2, stride=2, padding=0, bias=False)
        self.ln1b = torch.nn.LayerNorm([12,12,12])
        self.conv1c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.conv2a = torch.nn.Conv3d(1, ndf, kernel_size=2, stride=2, padding=0, bias=False)
        self.ln2a = torch.nn.LayerNorm([24,24,24])
        self.conv2b = torch.nn.Conv3d(ndf, ndf, kernel_size=2, stride=2, padding=0, bias=False)
        self.ln2b = torch.nn.LayerNorm([12,12,12])
        self.conv2c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False)

        self.conv_lin = torch.nn.Linear(6*6*6*ndf, 64) 
        self.conv_log = torch.nn.Linear(6*6*6*ndf, 64) 
        
        self.econd_lin = torch.nn.Linear(1, 64) 

        self.fc1 = torch.nn.Linear(64*7, 256)
        self.fc2 = torch.nn.Linear(256,  256)
        self.fc3 = torch.nn.Linear(256, 1)


    def forward(self, img, E_true):
        batch_size = img.size(0)
        
        #img_log_a = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)
        #img_log_b = img[:,0:1,:,:,:] #data.view(-1,1,30,30)

        #img_a = torch.exp(img_log_a-5.0)-np.exp(-5.0)
        #img_b = torch.exp(img_log_b-5.0)-np.exp(-5.0)
        
        
        img_a = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)
        img_b = img[:,0:1,:,:,:] #data.view(-1,1,30,30)
        
        img_log_a = torch.log(img_a+np.exp(-5.0))+5.0
        img_log_b = torch.log(img_b+np.exp(-5.0))+5.0

        
        img_a_hit_std = torch.std(img_a.view(batch_size, -1), 1).view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_a_hit_sum = torch.sum(img_a.view(batch_size, -1), 1).view(1, batch_size, 1).repeat(batch_size, 1, 1)*0.01
        
        img_a_hit_std = torch.transpose(img_a_hit_std, 1, 2)
        img_a_hit_sum = torch.transpose(img_a_hit_sum, 1, 2)

        
        img_a_hit_std = F.leaky_relu(self.Embed1a(img_a_hit_std))
        img_a_hit_std = F.leaky_relu(self.Embed1b(img_a_hit_std))
        img_a_hit_std = F.leaky_relu(self.Embed1c(img_a_hit_std))
        
        img_a_hit_sum = F.leaky_relu(self.Embed2a(img_a_hit_sum))
        img_a_hit_sum = F.leaky_relu(self.Embed2b(img_a_hit_sum))
        img_a_hit_sum = F.leaky_relu(self.Embed2c(img_a_hit_sum))

        img_a_hit_std_std  =  torch.std(img_a_hit_std, 2)
        img_a_hit_std_mean = torch.mean(img_a_hit_std, 2)

        img_a_hit_sum_std  =  torch.std(img_a_hit_sum, 2)
        img_a_hit_sum_mean = torch.mean(img_a_hit_sum, 2)
        
        xa1 = torch.cat((img_a_hit_std_std, img_a_hit_std_mean, img_a_hit_sum_std, img_a_hit_sum_mean), 1)
        xa1 = F.leaky_relu(self.embed_lin(xa1), 0.2)
        
        
        img_log_a_hit_std = torch.std(img_log_a.view(batch_size, -1), 1).view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_log_a_hit_sum = torch.sum(img_log_a.view(batch_size, -1), 1).view(1, batch_size, 1).repeat(batch_size, 1, 1)*0.01
        
        img_log_a_hit_std = torch.transpose(img_log_a_hit_std, 1, 2)
        img_log_a_hit_sum = torch.transpose(img_log_a_hit_sum, 1, 2)
        
        img_log_a_hit_std = F.leaky_relu(self.Embed1a(img_log_a_hit_std))
        img_log_a_hit_std = F.leaky_relu(self.Embed1b(img_log_a_hit_std))
        img_log_a_hit_std = F.leaky_relu(self.Embed1c(img_log_a_hit_std))
        
        img_log_a_hit_sum = F.leaky_relu(self.Embed2a(img_log_a_hit_sum))
        img_log_a_hit_sum = F.leaky_relu(self.Embed2b(img_log_a_hit_sum))
        img_log_a_hit_sum = F.leaky_relu(self.Embed2c(img_log_a_hit_sum))

        img_log_a_hit_std_std  =  torch.std(img_log_a_hit_std, 2)
        img_log_a_hit_std_mean = torch.mean(img_log_a_hit_std, 2)

        img_log_a_hit_sum_std  =  torch.std(img_log_a_hit_sum, 2)
        img_log_a_hit_sum_mean = torch.mean(img_log_a_hit_sum, 2)        
        
        xa2 = torch.cat((img_log_a_hit_std_std, img_log_a_hit_std_mean, img_log_a_hit_sum_std, img_log_a_hit_sum_mean), 1)
        xa2 = F.leaky_relu(self.embed_lin(xa2), 0.2)
        
               
        xa3 = F.leaky_relu(self.diff_lin((img_a-img_b).view(batch_size, -1)), 0.2)
        
        
        xa4 = F.leaky_relu(self.diff_log((img_log_a-img_log_b).view(batch_size, -1)), 0.2)
 

        xa5 = F.leaky_relu(self.ln1a(self.conv1a(img_a)), 0.2)
        xa5 = F.leaky_relu(self.ln1b(self.conv1b(xa5)), 0.2)
        xa5 = F.leaky_relu(self.conv1c(xa5), 0.2)
        xa5 = xa5.view(-1, self.ndf*6*6*6)
        xa5 = F.leaky_relu(self.conv_lin(xa5), 0.2)
        

        xa6 = F.leaky_relu(self.ln2a(self.conv2a(img_log_a)), 0.2)
        xa6 = F.leaky_relu(self.ln2b(self.conv2b(xa6)), 0.2)
        xa6 = F.leaky_relu(self.conv2c(xa6), 0.2)
        xa6 = xa6.view(-1, self.ndf*6*6*6)
        xa6 = F.leaky_relu(self.conv_log(xa6), 0.2)        
        
        xa7 = F.leaky_relu(self.econd_lin(E_true), 0.2)      
        
        xa = torch.cat((xa1, xa2, xa3, xa4, xa5, xa6, xa7), 1)
        
        xa = F.leaky_relu(self.fc1(xa), 0.2)
        xa = F.leaky_relu(self.fc2(xa), 0.2)
        xa = self.fc3(xa)

        return xa.view(-1) ### flattens
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
class BiBAE_F_3D_BatchStat_n16(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, device, nc=1, ngf=16, z_rand=500, z_enc=12):
        super(BiBAE_F_3D_BatchStat_n16, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        self.z_rand = z_rand
        self.z_enc = z_enc
        self.z_full = z_enc + z_rand
        self.args = args
        self.device = device

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(2,2,2), stride=(2,2,2),
                               padding=(0,0,0), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.LayerNorm([24,24,24])
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(2,2,2), stride=(2,2,2),
                               padding=(0,0,0), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.LayerNorm([12,12,12])
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.LayerNorm([6,6,6])
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*4, kernel_size=(4,4,4), stride=(1,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.LayerNorm([5,5,5])

     
        self.fc1 = nn.Linear(5*5*5*ngf*4+1, ngf*100, bias=True)
        self.fc2 = nn.Linear(ngf*100, int(self.z_full), bias=True)
        
        self.fc31 = nn.Linear(int(self.z_full), self.z_enc, bias=True)
        self.fc32 = nn.Linear(int(self.z_full), self.z_enc, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z_full+1, int(self.z_full), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z_full), ngf*100, bias=True)
        self.cond3 = torch.nn.Linear(ngf*100, 12*12*12*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.LayerNorm([24,24,24])
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.LayerNorm([48,48,48])
        
        self.conv0 = torch.nn.Conv3d(ngf, ngf, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco0 = torch.nn.LayerNorm([48,48,48])
        self.conv1 = torch.nn.Conv3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco1 = torch.nn.LayerNorm([48,48,48])
        self.conv2 = torch.nn.Conv3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco2 = torch.nn.LayerNorm([48,48,48])
        self.conv3 = torch.nn.Conv3d(ngf*4, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.bnco3 = torch.nn.LayerNorm([48,48,48])
        self.conv4 = torch.nn.Conv3d(ngf*2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)


    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,48,48,48))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        return torch.cat((self.fc31(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1), torch.cat((self.fc32(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)

        x = x.view(-1,self.ngf,12,12,12)        

        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 24, 24, 24])), 0.2, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 48, 48, 48])), 0.2, inplace=True) #

        x = F.leaky_relu(self.bnco0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco3(self.conv3(x)), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        return x 
        
        
    def forward(self, x, E_true, z=None, mode='full'):
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z 
        elif mode == 'decode':
            return self.decode(torch.cat((z,E_true), 1)) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), mu, logvar, z
            
        

    
    
    
    
    

    
    
    
    
class Discriminator_F_BatchStatV2_LogScale(nn.Module):
    def __init__(self, isize=48, nc=2, ndf=64):
        super(Discriminator_F_BatchStatV2_LogScale, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc
        self.size_embed = 16
        self.conv1_bias = False

        self.Embed1a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed1b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed1c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed2a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed2b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed2c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed3a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed3b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed3c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed4a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed4b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed4c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        
        self.embed_lin = torch.nn.Linear(self.size_embed*4, 64) 
        self.embed_log = torch.nn.Linear(self.size_embed*4, 64) 
        self.diff_lin = torch.nn.Linear(48*48*48, 64) 
        self.diff_log = torch.nn.Linear(48*48*48, 64) 
        
        
        self.conv1a = torch.nn.Conv3d(1, ndf, kernel_size=2, stride=2, padding=0, bias=False)
        self.ln1a = torch.nn.LayerNorm([24,24,24])
        self.conv1b = torch.nn.Conv3d(ndf, ndf, kernel_size=2, stride=2, padding=0, bias=False)
        self.ln1b = torch.nn.LayerNorm([12,12,12])
        self.conv1c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.conv2a = torch.nn.Conv3d(1, ndf, kernel_size=2, stride=2, padding=0, bias=False)
        self.ln2a = torch.nn.LayerNorm([24,24,24])
        self.conv2b = torch.nn.Conv3d(ndf, ndf, kernel_size=2, stride=2, padding=0, bias=False)
        self.ln2b = torch.nn.LayerNorm([12,12,12])
        self.conv2c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False)

        self.conv_lin = torch.nn.Linear(6*6*6*ndf, 64) 
        self.conv_log = torch.nn.Linear(6*6*6*ndf, 64) 
        
        self.econd_lin = torch.nn.Linear(1, 64) 

        self.fc1 = torch.nn.Linear(64*7, 256)
        self.fc2 = torch.nn.Linear(256,  256)
        self.fc3 = torch.nn.Linear(256, 1)


    def forward(self, img, E_true):
        batch_size = img.size(0)
        
        img_log_a = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)
        img_log_b = img[:,0:1,:,:,:] #data.view(-1,1,30,30)

        #print('img_log_a', torch.sum(img_log_a))
        #print('img_log_b', torch.sum(img_log_b))

        img_a = torch.exp(img_log_a-5.0)-np.exp(-5.0)
        img_b = torch.exp(img_log_b-5.0)-np.exp(-5.0)
        
        
        img_a_hit_std = torch.std(img_a.view(batch_size, -1), 1)
        img_a_hit_std_a = img_a_hit_std.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_a_hit_std_b = img_a_hit_std.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_a_hit_std = img_a_hit_std_a - img_a_hit_std_b
        
        img_a_hit_sum = torch.sum(img_a.view(batch_size, -1), 1)
        img_a_hit_sum_a = img_a_hit_sum.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_a_hit_sum_b = img_a_hit_sum.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_a_hit_sum = img_a_hit_sum_a - img_a_hit_sum_b

        
        img_a_hit_std = torch.transpose(img_a_hit_std, 1, 2)
        img_a_hit_sum = torch.transpose(img_a_hit_sum, 1, 2)

        
        img_a_hit_std = F.leaky_relu(self.Embed1a(img_a_hit_std))
        img_a_hit_std = F.leaky_relu(self.Embed1b(img_a_hit_std))
        img_a_hit_std = F.leaky_relu(self.Embed1c(img_a_hit_std))
        
        img_a_hit_sum = F.leaky_relu(self.Embed2a(img_a_hit_sum))
        img_a_hit_sum = F.leaky_relu(self.Embed2b(img_a_hit_sum))
        img_a_hit_sum = F.leaky_relu(self.Embed2c(img_a_hit_sum))

        img_a_hit_std_std  =  torch.std(img_a_hit_std, 2)
        img_a_hit_std_mean = torch.mean(img_a_hit_std, 2)

        img_a_hit_sum_std  =  torch.std(img_a_hit_sum, 2)
        img_a_hit_sum_mean = torch.mean(img_a_hit_sum, 2)
        
        xa1 = torch.cat((img_a_hit_std_std, img_a_hit_std_mean, img_a_hit_sum_std, img_a_hit_sum_mean), 1)
        xa1 = F.leaky_relu(self.embed_lin(xa1), 0.2)
        
        
        img_log_a_hit_std = torch.std(img_log_a.view(batch_size, -1), 1)
        img_log_a_hit_std_a = img_log_a_hit_std.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_log_a_hit_std_b = img_log_a_hit_std.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_log_a_hit_std = img_log_a_hit_std_a - img_log_a_hit_std_b
        
        img_log_a_hit_sum = torch.sum(img_log_a.view(batch_size, -1), 1)
        img_log_a_hit_sum_a = img_log_a_hit_sum.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_log_a_hit_sum_b = img_log_a_hit_sum.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_log_a_hit_sum = img_log_a_hit_sum_a - img_log_a_hit_sum_b
        
        img_log_a_hit_std = torch.transpose(img_log_a_hit_std, 1, 2)
        img_log_a_hit_sum = torch.transpose(img_log_a_hit_sum, 1, 2)
        
        img_log_a_hit_std = F.leaky_relu(self.Embed1a(img_log_a_hit_std))
        img_log_a_hit_std = F.leaky_relu(self.Embed1b(img_log_a_hit_std))
        img_log_a_hit_std = F.leaky_relu(self.Embed1c(img_log_a_hit_std))
        
        img_log_a_hit_sum = F.leaky_relu(self.Embed2a(img_log_a_hit_sum))
        img_log_a_hit_sum = F.leaky_relu(self.Embed2b(img_log_a_hit_sum))
        img_log_a_hit_sum = F.leaky_relu(self.Embed2c(img_log_a_hit_sum))

        img_log_a_hit_std_std  =  torch.std(img_log_a_hit_std, 2)
        img_log_a_hit_std_mean = torch.mean(img_log_a_hit_std, 2)

        img_log_a_hit_sum_std  =  torch.std(img_log_a_hit_sum, 2)
        img_log_a_hit_sum_mean = torch.mean(img_log_a_hit_sum, 2)        
        
        xa2 = torch.cat((img_log_a_hit_std_std, img_log_a_hit_std_mean, img_log_a_hit_sum_std, img_log_a_hit_sum_mean), 1)
        xa2 = F.leaky_relu(self.embed_lin(xa2), 0.2)
        
               
        xa3 = F.leaky_relu(self.diff_lin((img_a-img_b).view(batch_size, -1)), 0.2)
        
        
        xa4 = F.leaky_relu(self.diff_log((img_log_a-img_log_b).view(batch_size, -1)), 0.2)
 

        xa5 = F.leaky_relu(self.ln1a(self.conv1a(img_a*0.001)), 0.2)
        xa5 = F.leaky_relu(self.ln1b(self.conv1b(xa5)), 0.2)
        xa5 = F.leaky_relu(self.conv1c(xa5), 0.2)
        xa5 = xa5.view(-1, self.ndf*6*6*6)
        xa5 = F.leaky_relu(self.conv_lin(xa5), 0.2)
        

        xa6 = F.leaky_relu(self.ln2a(self.conv2a(img_log_a)), 0.2)
        xa6 = F.leaky_relu(self.ln2b(self.conv2b(xa6)), 0.2)
        xa6 = F.leaky_relu(self.conv2c(xa6), 0.2)
        xa6 = xa6.view(-1, self.ndf*6*6*6)
        xa6 = F.leaky_relu(self.conv_log(xa6), 0.2)        
        
        xa7 = F.leaky_relu(self.econd_lin(E_true), 0.2)      
        
        xa = torch.cat((xa1, xa2, xa3, xa4, xa5, xa6, xa7), 1)
        
        #print('xa1', torch.sum(xa1))
        #print('xa2', torch.sum(xa2))
        #print('xa3', torch.sum(xa3))
        #print('xa4', torch.sum(xa4))
        #print('xa5', torch.sum(xa5))
        #print('xa6', torch.sum(xa6))
        #print('xa7', torch.sum(xa7))
        
        xa = F.leaky_relu(self.fc1(xa), 0.2)
        xa = F.leaky_relu(self.fc2(xa), 0.2)
        xa = self.fc3(xa)

        #print('xa', torch.sum(xa))
        
        return xa ### flattens
    
    
    
    
    
    
    

    
    
    
    
    
    

    
    
    
    
class Discriminator_F_BatchStatV3_LogScale(nn.Module):
    def __init__(self, isize=48, nc=2, ndf=64):
        super(Discriminator_F_BatchStatV3_LogScale, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc
        self.size_embed = 16
        self.conv1_bias = False

        self.Embed1a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed1b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed1c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed2a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed2b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed2c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed3a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed3b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed3c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed4a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed4b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed4c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        
        self.embed_lin = torch.nn.Linear(self.size_embed*4, 64) 
        self.embed_log = torch.nn.Linear(self.size_embed*4, 64) 
        self.diff_lin = torch.nn.Linear(48*48*48, 64) 
        self.diff_log = torch.nn.Linear(48*48*48, 64) 
        
        
        self.conv1a = torch.nn.Conv3d(1, ndf, kernel_size=2, stride=2, padding=0, bias=False)
        self.ln1a = torch.nn.LayerNorm([24,24,24])
        self.conv1b = torch.nn.Conv3d(ndf, ndf, kernel_size=2, stride=2, padding=0, bias=False)
        self.ln1b = torch.nn.LayerNorm([12,12,12])
        self.conv1c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.conv2a = torch.nn.Conv3d(1, ndf, kernel_size=2, stride=2, padding=0, bias=False)
        self.ln2a = torch.nn.LayerNorm([24,24,24])
        self.conv2b = torch.nn.Conv3d(ndf, ndf, kernel_size=2, stride=2, padding=0, bias=False)
        self.ln2b = torch.nn.LayerNorm([12,12,12])
        self.conv2c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False)

        self.conv_lin = torch.nn.Linear(6*6*6*ndf, 64) 
        self.conv_log = torch.nn.Linear(6*6*6*ndf, 64) 
        
        self.econd_lin = torch.nn.Linear(1, 64) 

        self.fc1 = torch.nn.Linear(64*7, 256)
        self.fc2 = torch.nn.Linear(256,  256)
        self.fc3 = torch.nn.Linear(256, 1)


    def forward(self, img, E_true):
        batch_size = img.size(0)
        im_size = img.size(-1)
        
        img_log_a = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)
        img_log_b = img[:,0:1,:,:,:] #data.view(-1,1,30,30)

        print('img_log_a', torch.sum(img_log_a))
        print('img_log_b', torch.sum(img_log_b))

        img_a = torch.exp(img_log_a-5.0)-np.exp(-5.0)
        img_b = torch.exp(img_log_b-5.0)-np.exp(-5.0)
        
        img_a_repeat1 = img_a.view(1, batch_size, 1, im_size, im_size, im_size).repeat(batch_size, 1, 1, 1, 1, 1)
        img_a_repeat2 = img_a.view(batch_size, 1, 1, im_size, im_size, im_size).repeat(1, batch_size, 1, 1, 1, 1)
        
        img_a_repeat = (img_a_repeat1 - img_a_repeat2) * 0.0
        
        img_a_hit_std = torch.std(img_a_repeat, (3,4,5))
        img_a_hit_sum = torch.sum(img_a_repeat, (3,4,5))
        
        img_a_hit_std = torch.transpose(img_a_hit_std, 1, 2)
        img_a_hit_sum = torch.transpose(img_a_hit_sum, 1, 2)

        img_a_hit_std = F.leaky_relu(self.Embed1a(img_a_hit_std))
        img_a_hit_std = F.leaky_relu(self.Embed1b(img_a_hit_std))
        img_a_hit_std = F.leaky_relu(self.Embed1c(img_a_hit_std))
        
        img_a_hit_sum = F.leaky_relu(self.Embed2a(img_a_hit_sum))
        img_a_hit_sum = F.leaky_relu(self.Embed2b(img_a_hit_sum))
        img_a_hit_sum = F.leaky_relu(self.Embed2c(img_a_hit_sum))

        img_a_hit_std_std  =  torch.std(img_a_hit_std, 2)
        img_a_hit_std_mean = torch.mean(img_a_hit_std, 2)

        img_a_hit_sum_std  =  torch.std(img_a_hit_sum, 2)
        img_a_hit_sum_mean = torch.mean(img_a_hit_sum, 2)
        
        xa1 = torch.cat((img_a_hit_std_std, img_a_hit_std_mean, img_a_hit_sum_std, img_a_hit_sum_mean), 1)
        xa1 = F.leaky_relu(self.embed_lin(xa1), 0.2)
        
        
        img_log_a_repeat1 = img_log_a.view(1, batch_size, 1, im_size, im_size, im_size).repeat(batch_size, 1, 1, 1, 1, 1)
        img_log_a_repeat2 = img_log_a.view(batch_size, 1, 1, im_size, im_size, im_size).repeat(1, batch_size, 1, 1, 1, 1)
        
        img_log_a_repeat = (img_log_a_repeat1 - img_log_a_repeat2) * 0.0
        
        img_log_a_hit_std = torch.std(img_log_a_repeat, (3,4,5))
        img_log_a_hit_sum = torch.sum(img_log_a_repeat, (3,4,5))
        
        img_log_a_hit_std = torch.transpose(img_log_a_hit_std, 1, 2)
        img_log_a_hit_sum = torch.transpose(img_log_a_hit_sum, 1, 2)
        
        img_log_a_hit_std = F.leaky_relu(self.Embed1a(img_log_a_hit_std))
        img_log_a_hit_std = F.leaky_relu(self.Embed1b(img_log_a_hit_std))
        img_log_a_hit_std = F.leaky_relu(self.Embed1c(img_log_a_hit_std))
        
        img_log_a_hit_sum = F.leaky_relu(self.Embed2a(img_log_a_hit_sum))
        img_log_a_hit_sum = F.leaky_relu(self.Embed2b(img_log_a_hit_sum))
        img_log_a_hit_sum = F.leaky_relu(self.Embed2c(img_log_a_hit_sum))

        img_log_a_hit_std_std  =  torch.std(img_log_a_hit_std, 2)
        img_log_a_hit_std_mean = torch.mean(img_log_a_hit_std, 2)

        img_log_a_hit_sum_std  =  torch.std(img_log_a_hit_sum, 2)
        img_log_a_hit_sum_mean = torch.mean(img_log_a_hit_sum, 2)        
        
        xa2 = torch.cat((img_log_a_hit_std_std, img_log_a_hit_std_mean, img_log_a_hit_sum_std, img_log_a_hit_sum_mean), 1)
        xa2 = F.leaky_relu(self.embed_lin(xa2), 0.2)
        
               
        xa3 = F.leaky_relu(self.diff_lin((img_a-img_b).view(batch_size, -1)), 0.2)
        
        
        xa4 = F.leaky_relu(self.diff_log((img_log_a-img_log_b).view(batch_size, -1)), 0.2)
 

        xa5 = F.leaky_relu(self.ln1a(self.conv1a(img_a*0.001)), 0.2)
        xa5 = F.leaky_relu(self.ln1b(self.conv1b(xa5)), 0.2)
        xa5 = F.leaky_relu(self.conv1c(xa5), 0.2)
        xa5 = xa5.view(-1, self.ndf*6*6*6)
        xa5 = F.leaky_relu(self.conv_lin(xa5), 0.2)
        

        xa6 = F.leaky_relu(self.ln2a(self.conv2a(img_log_a)), 0.2)
        xa6 = F.leaky_relu(self.ln2b(self.conv2b(xa6)), 0.2)
        xa6 = F.leaky_relu(self.conv2c(xa6), 0.2)
        xa6 = xa6.view(-1, self.ndf*6*6*6)
        xa6 = F.leaky_relu(self.conv_log(xa6), 0.2)        
        
        xa7 = F.leaky_relu(self.econd_lin(E_true), 0.2)      
        
        xa = torch.cat((xa1, xa2, xa3, xa4, xa5, xa6, xa7), 1)
        
        print('xa1', torch.sum(xa1))
        print('xa2', torch.sum(xa2))
        print('xa3', torch.sum(xa3))
        print('xa4', torch.sum(xa4))
        print('xa5', torch.sum(xa5))
        print('xa6', torch.sum(xa6))
        print('xa7', torch.sum(xa7))
        
        xa = F.leaky_relu(self.fc1(xa), 0.2)
        xa = F.leaky_relu(self.fc2(xa), 0.2)
        xa = self.fc3(xa)

        #print('xa', torch.sum(xa))
        
        return xa ### flattens
    
    
    
    
    
    
    
#######################################################################################################################
######################################################## Core #########################################################
#######################################################################################################################


    
class BiBAE_F_3D_BatchStat_Core(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, device, nc=1, ngf=8, z_rand=500, z_enc=12):
        super(BiBAE_F_3D_BatchStat_Core, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        self.z_rand = z_rand
        self.z_enc = z_enc
        self.z_full = z_enc + z_rand
        self.args = args
        self.device = device

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(2,2,2), stride=(2,1,1),
                               padding=(0,0,0), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.LayerNorm([24,11,11])
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(2,2,2), stride=(2,1,1),
                               padding=(0,0,0), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.LayerNorm([12,10,10])
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.LayerNorm([6,5,5])
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*4, kernel_size=(4,4,4), stride=(1,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.LayerNorm([5,4,4])

     
        self.fc1 = nn.Linear(5*4*4*ngf*4+1, ngf*100, bias=True)
        self.fc2 = nn.Linear(ngf*100, int(self.z_full), bias=True)
        
        self.fc31 = nn.Linear(int(self.z_full), self.z_enc, bias=True)
        self.fc32 = nn.Linear(int(self.z_full), self.z_enc, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z_full+1, int(self.z_full), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z_full), ngf*100, bias=True)
        self.cond3 = torch.nn.Linear(ngf*100, 6*6*12*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.LayerNorm([24,12,12])
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.LayerNorm([48,24,24])

        self.conv0a = torch.nn.Conv3d(ngf, ngf, kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0), bias=False)
        self.bnco0a = torch.nn.LayerNorm([48,12,12])
        self.conv0 = torch.nn.Conv3d(ngf, ngf, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco0 = torch.nn.LayerNorm([48,12,12])
        self.conv1 = torch.nn.Conv3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco1 = torch.nn.LayerNorm([48,12,12])
        self.conv2 = torch.nn.Conv3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco2 = torch.nn.LayerNorm([48,12,12])
        self.conv3 = torch.nn.Conv3d(ngf*4, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.bnco3 = torch.nn.LayerNorm([48,12,12])
        self.conv4 = torch.nn.Conv3d(ngf*2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)


    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,48,12,12))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        return torch.cat((self.fc31(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1), torch.cat((self.fc32(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)

        x = x.view(-1,self.ngf,12,6,6)        

        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 24, 12, 12])), 0.2, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 48, 24, 24])), 0.2, inplace=True) #


        x = F.leaky_relu(self.bnco0a(self.conv0a(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco3(self.conv3(x)), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        return x 
        
        
    def forward(self, x, E_true, z=None, mode='full'):
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z 
        elif mode == 'decode':
            return self.decode(torch.cat((z,E_true), 1)) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), mu, logvar, z
        elif mode == 'half':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1))
            
    
    

    
    
    
    
    
class Discriminator_F_BatchStatV2_Core(nn.Module):
    def __init__(self, isize=48, nc=2, ndf=64):
        super(Discriminator_F_BatchStatV2_Core, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc
        self.size_embed = 16
        self.conv1_bias = False

        self.Embed1a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed1b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed1c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed2a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed2b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed2c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed3a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed3b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed3c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed4a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed4b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed4c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        
        self.embed_lin = torch.nn.Linear(self.size_embed*4, 64) 
        self.embed_log = torch.nn.Linear(self.size_embed*4, 64) 
        self.diff_lin = torch.nn.Linear(48*12*12, 64) 
        self.diff_log = torch.nn.Linear(48*12*12, 64) 
        
        
        self.conv1a = torch.nn.Conv3d(1, ndf, kernel_size=2, stride=(2,1,1), padding=0, bias=False)
        self.ln1a = torch.nn.LayerNorm([24,11,11])
        self.conv1b = torch.nn.Conv3d(ndf, ndf, kernel_size=2, stride=(2,1,1), padding=0, bias=False)
        self.ln1b = torch.nn.LayerNorm([12,10,10])
        self.conv1c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=(2,2,2), padding=1, bias=False)
        
        self.conv2a = torch.nn.Conv3d(1, ndf, kernel_size=2, stride=(2,1,1), padding=0, bias=False)
        self.ln2a = torch.nn.LayerNorm([24,11,11])
        self.conv2b = torch.nn.Conv3d(ndf, ndf, kernel_size=2, stride=(2,1,1), padding=0, bias=False)
        self.ln2b = torch.nn.LayerNorm([12,10,10])
        self.conv2c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=(2,2,2), padding=1, bias=False)

        self.conv_lin = torch.nn.Linear(6*5*5*ndf, 64) 
        self.conv_log = torch.nn.Linear(6*5*5*ndf, 64) 
        
        self.econd_lin = torch.nn.Linear(1, 64) 

        self.fc1 = torch.nn.Linear(64*7, 256)
        self.fc2 = torch.nn.Linear(256,  256)
        self.fc3 = torch.nn.Linear(256, 1)


    def forward(self, img, E_true):
        batch_size = img.size(0)
        
        #img_log_a = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)
        #img_log_b = img[:,0:1,:,:,:] #data.view(-1,1,30,30)

        #print('img_log_a', torch.sum(img_log_a))
        #print('img_log_b', torch.sum(img_log_b))

        #img_a = torch.exp(img_log_a-5.0)-np.exp(-5.0)
        #img_b = torch.exp(img_log_b-5.0)-np.exp(-5.0)
        
        img_a = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)
        img_b = img[:,0:1,:,:,:] #data.view(-1,1,30,30)
        
        img_log_a = torch.log(img_a+np.exp(-5.0))+5.0
        img_log_b = torch.log(img_b+np.exp(-5.0))+5.0
        
        
        img_a_hit_std = torch.std(img_a.view(batch_size, -1), 1)
        img_a_hit_std_a = img_a_hit_std.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_a_hit_std_b = img_a_hit_std.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_a_hit_std = img_a_hit_std_a - img_a_hit_std_b
        
        img_a_hit_sum = torch.sum(img_a.view(batch_size, -1), 1)
        img_a_hit_sum_a = img_a_hit_sum.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_a_hit_sum_b = img_a_hit_sum.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_a_hit_sum = img_a_hit_sum_a - img_a_hit_sum_b

        
        img_a_hit_std = torch.transpose(img_a_hit_std, 1, 2)
        img_a_hit_sum = torch.transpose(img_a_hit_sum, 1, 2)

        
        img_a_hit_std = F.leaky_relu(self.Embed1a(img_a_hit_std))
        img_a_hit_std = F.leaky_relu(self.Embed1b(img_a_hit_std))
        img_a_hit_std = F.leaky_relu(self.Embed1c(img_a_hit_std))
        
        img_a_hit_sum = F.leaky_relu(self.Embed2a(img_a_hit_sum))
        img_a_hit_sum = F.leaky_relu(self.Embed2b(img_a_hit_sum))
        img_a_hit_sum = F.leaky_relu(self.Embed2c(img_a_hit_sum))

        img_a_hit_std_std  =  torch.std(img_a_hit_std, 2)
        img_a_hit_std_mean = torch.mean(img_a_hit_std, 2)

        img_a_hit_sum_std  =  torch.std(img_a_hit_sum, 2)
        img_a_hit_sum_mean = torch.mean(img_a_hit_sum, 2)
        
        xa1 = torch.cat((img_a_hit_std_std, img_a_hit_std_mean, img_a_hit_sum_std, img_a_hit_sum_mean), 1)
        xa1 = F.leaky_relu(self.embed_lin(xa1), 0.2)
        
        
        img_log_a_hit_std = torch.std(img_log_a.view(batch_size, -1), 1)
        img_log_a_hit_std_a = img_log_a_hit_std.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_log_a_hit_std_b = img_log_a_hit_std.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_log_a_hit_std = img_log_a_hit_std_a - img_log_a_hit_std_b
        
        img_log_a_hit_sum = torch.sum(img_log_a.view(batch_size, -1), 1)
        img_log_a_hit_sum_a = img_log_a_hit_sum.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_log_a_hit_sum_b = img_log_a_hit_sum.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_log_a_hit_sum = img_log_a_hit_sum_a - img_log_a_hit_sum_b
        
        img_log_a_hit_std = torch.transpose(img_log_a_hit_std, 1, 2)
        img_log_a_hit_sum = torch.transpose(img_log_a_hit_sum, 1, 2)
        
        img_log_a_hit_std = F.leaky_relu(self.Embed1a(img_log_a_hit_std))
        img_log_a_hit_std = F.leaky_relu(self.Embed1b(img_log_a_hit_std))
        img_log_a_hit_std = F.leaky_relu(self.Embed1c(img_log_a_hit_std))
        
        img_log_a_hit_sum = F.leaky_relu(self.Embed2a(img_log_a_hit_sum))
        img_log_a_hit_sum = F.leaky_relu(self.Embed2b(img_log_a_hit_sum))
        img_log_a_hit_sum = F.leaky_relu(self.Embed2c(img_log_a_hit_sum))

        img_log_a_hit_std_std  =  torch.std(img_log_a_hit_std, 2)
        img_log_a_hit_std_mean = torch.mean(img_log_a_hit_std, 2)

        img_log_a_hit_sum_std  =  torch.std(img_log_a_hit_sum, 2)
        img_log_a_hit_sum_mean = torch.mean(img_log_a_hit_sum, 2)        
        
        xa2 = torch.cat((img_log_a_hit_std_std, img_log_a_hit_std_mean, img_log_a_hit_sum_std, img_log_a_hit_sum_mean), 1)
        xa2 = F.leaky_relu(self.embed_lin(xa2), 0.2)
        
               
        xa3 = F.leaky_relu(self.diff_lin((img_a-img_b).view(batch_size, -1)), 0.2)
        
        
        xa4 = F.leaky_relu(self.diff_log((img_log_a-img_log_b).view(batch_size, -1)), 0.2)
 

        xa5 = F.leaky_relu(self.ln1a(self.conv1a(img_a*0.001)), 0.2)
        xa5 = F.leaky_relu(self.ln1b(self.conv1b(xa5)), 0.2)
        xa5 = F.leaky_relu(self.conv1c(xa5), 0.2)
        xa5 = xa5.view(-1, self.ndf*6*5*5)
        xa5 = F.leaky_relu(self.conv_lin(xa5), 0.2)
        

        xa6 = F.leaky_relu(self.ln2a(self.conv2a(img_log_a)), 0.2)
        xa6 = F.leaky_relu(self.ln2b(self.conv2b(xa6)), 0.2)
        xa6 = F.leaky_relu(self.conv2c(xa6), 0.2)
        xa6 = xa6.view(-1, self.ndf*6*5*5)
        xa6 = F.leaky_relu(self.conv_log(xa6), 0.2)        
        
        xa7 = F.leaky_relu(self.econd_lin(E_true), 0.2)      
        
        xa = torch.cat((xa1, xa2, xa3, xa4, xa5, xa6, xa7), 1)
        
        #print('xa1', torch.sum(xa1))
        #print('xa2', torch.sum(xa2))
        #print('xa3', torch.sum(xa3))
        #print('xa4', torch.sum(xa4))
        #print('xa5', torch.sum(xa5))
        #print('xa6', torch.sum(xa6))
        #print('xa7', torch.sum(xa7))
        
        xa = F.leaky_relu(self.fc1(xa), 0.2)
        xa = F.leaky_relu(self.fc2(xa), 0.2)
        xa = self.fc3(xa)

        #print('xa', torch.sum(xa))
        
        return xa ### flattens
    
    
    
    
    
class PostProcess_Size1Conv_EcondV2_Core(nn.Module):
    def __init__(self, isize=48, isize2=12, nc=2, ndf=64, bias=False, out_funct='relu'):
        super(PostProcess_Size1Conv_EcondV2_Core, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.isize2 = isize2
        self.nc = nc
        self.bais = bias
        self.out_funct = out_funct
        
        self.fcec1 = torch.nn.Linear(2, int(ndf/2), bias=True)
        self.fcec2 = torch.nn.Linear(int(ndf/2), int(ndf/2), bias=True)
        self.fcec3 = torch.nn.Linear(int(ndf/2), int(ndf/2), bias=True)
        
        self.conv1 = torch.nn.Conv3d(1, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco1 = torch.nn.LayerNorm([self.isize, self.isize2, self.isize2])
        
        self.conv2 = torch.nn.Conv3d(ndf+int(ndf/2), ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco2 = torch.nn.LayerNorm([self.isize, self.isize2, self.isize2])

        self.conv3 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco3 = torch.nn.LayerNorm([self.isize, self.isize2, self.isize2])

        self.conv4 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco4 = torch.nn.LayerNorm([self.isize, self.isize2, self.isize2])

        self.conv5 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)

        self.conv6 = torch.nn.Conv3d(ndf, 1, kernel_size=1, stride=1, padding=0, bias=False)
 

    def forward(self, img, E_True=0):
        img = img.view(-1, 1, self.isize, self.isize2, self.isize2) 
                
        econd = torch.cat((torch.sum(img.view(-1, self.isize2*self.isize2*self.isize), 1).view(-1, 1), E_True), 1)

        econd = F.leaky_relu(self.fcec1(econd), 0.01)
        econd = F.leaky_relu(self.fcec2(econd), 0.01)
        econd = F.leaky_relu(self.fcec3(econd), 0.01)
        
        econd = econd.view(-1, int(self.ndf/2), 1, 1, 1)
        econd = econd.expand(-1, -1, self.isize, self.isize2, self.isize2)        
        
        img = F.leaky_relu(self.bnco1(self.conv1(img)), 0.01)
        img = torch.cat((img, econd), 1)
        img = F.leaky_relu(self.bnco2(self.conv2(img)), 0.01)
        img = F.leaky_relu(self.bnco3(self.conv3(img)), 0.01)
        img = F.leaky_relu(self.bnco4(self.conv4(img)), 0.01)
        img = F.leaky_relu(self.conv5(img), 0.01) 
        img = self.conv6(img)

        if self.out_funct == 'relu':
            img = F.relu(img)
        elif self.out_funct == 'leaky_relu':
            img = F.leaky_relu(img, 0.01) 
              
        return img.view(-1, 1, self.isize, self.isize2, self.isize2)


     

class PostProcess_Lin_Core(nn.Module):
    def __init__(self, isize=48, isize2=12, nc=2, ndf=64, bias=False, out_funct='relu'):
        super(PostProcess_Lin_Core, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.isize2 = isize2
        self.nc = nc
        self.bais = bias
        self.out_funct = out_funct
        
        self.fc1 = torch.nn.Linear(1,   ndf, bias=bias)
        self.fc2 = torch.nn.Linear(ndf, ndf, bias=bias)
        self.fc3 = torch.nn.Linear(ndf, ndf, bias=bias)
        self.fc4 = torch.nn.Linear(ndf, ndf, bias=bias)
        self.fc5 = torch.nn.Linear(ndf, ndf, bias=bias)
        self.fc6 = torch.nn.Linear(ndf, 1,   bias=False)

 

    def forward(self, img, E_True=0):
        img = img.view(-1, self.isize, self.isize2, self.isize2, 1)
                
        img = F.leaky_relu(self.fc1(img), 0.01)
        img = F.leaky_relu(self.fc2(img), 0.01)
        img = F.leaky_relu(self.fc3(img), 0.01)
        img = F.leaky_relu(self.fc4(img), 0.01)
        img = F.leaky_relu(self.fc5(img), 0.01) 
        img = self.fc6(img)

        if self.out_funct == 'relu':
            img = F.relu(img)
        elif self.out_funct == 'leaky_relu':
            img = F.leaky_relu(img, 0.01) 
              
        return img.view(-1, 1, self.isize, self.isize2, self.isize2)
    
    
    
    
    
    
    
class BiBAE_F_3D_BatchStat_Core_V2(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, device, nc=1, ngf=8, z_rand=500, z_enc=12):
        super(BiBAE_F_3D_BatchStat_Core_V2, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        self.z_rand = z_rand
        self.z_enc = z_enc
        self.z_full = z_enc + z_rand
        self.args = args
        self.device = device

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(2,2,2), stride=(2,1,1),
                               padding=(0,0,0), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.LayerNorm([24,11,11])
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(2,2,2), stride=(2,1,1),
                               padding=(0,0,0), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.LayerNorm([12,10,10])
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.LayerNorm([6,5,5])
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*4, kernel_size=(4,4,4), stride=(1,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.LayerNorm([5,4,4])

     
        self.fc1 = nn.Linear(5*4*4*ngf*4+1, ngf*100, bias=True)
        self.fc2 = nn.Linear(ngf*100, int(self.z_full), bias=True)
        self.fc3 = nn.Linear(int(self.z_full), int(self.z_full), bias=True)
        self.fc4 = nn.Linear(int(self.z_full), int(self.z_full), bias=True)
        self.fc5 = nn.Linear(int(self.z_full), int(self.z_full), bias=True)
        
        self.fc61 = nn.Linear(int(self.z_full), self.z_enc, bias=True)
        self.fc62 = nn.Linear(int(self.z_full), self.z_enc, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z_full+1, int(self.z_full), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z_full), int(self.z_full), bias=True)
        self.cond3 = torch.nn.Linear(int(self.z_full), int(self.z_full), bias=True)
        self.cond4 = torch.nn.Linear(int(self.z_full), ngf*100, bias=True)
        self.cond5 = torch.nn.Linear(ngf*100, 6*6*12*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.LayerNorm([24,12,12])
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.LayerNorm([48,24,24])

        self.conv0a = torch.nn.Conv3d(ngf, ngf, kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0), bias=False)
        self.bnco0a = torch.nn.LayerNorm([48,12,12])
        self.conv0 = torch.nn.Conv3d(ngf, ngf, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco0 = torch.nn.LayerNorm([48,12,12])
        self.conv1 = torch.nn.Conv3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco1 = torch.nn.LayerNorm([48,12,12])
        self.conv2 = torch.nn.Conv3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco2 = torch.nn.LayerNorm([48,12,12])
        self.conv3 = torch.nn.Conv3d(ngf*4, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.bnco3 = torch.nn.LayerNorm([48,12,12])
        self.conv4 = torch.nn.Conv3d(ngf*2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)


    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,48,12,12))), 0.02, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.02, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.02, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.02, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu((self.fc1(x)), 0.02, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.02, inplace=True)
        x = F.leaky_relu((self.fc3(x)), 0.02, inplace=True)
        x = F.leaky_relu((self.fc4(x)), 0.02, inplace=True)
        x = F.leaky_relu((self.fc5(x)), 0.02, inplace=True)
        return torch.cat((self.fc61(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1), torch.cat((self.fc62(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        x = F.leaky_relu((self.cond1(z)), 0.02, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.02, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.02, inplace=True)
        x = F.leaky_relu((self.cond4(x)), 0.02, inplace=True)
        x = F.leaky_relu((self.cond5(x)), 0.02, inplace=True)

        x = x.view(-1,self.ngf,12,6,6)        

        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 24, 12, 12])), 0.02, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 48, 24, 24])), 0.02, inplace=True) #


        x = F.leaky_relu(self.bnco0a(self.conv0a(x)), 0.02, inplace=True)
        x = F.leaky_relu(self.bnco0(self.conv0(x)), 0.02, inplace=True)
        x = F.leaky_relu(self.bnco1(self.conv1(x)), 0.02, inplace=True)
        x = F.leaky_relu(self.bnco2(self.conv2(x)), 0.02, inplace=True)
        x = F.leaky_relu(self.bnco3(self.conv3(x)), 0.02, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        return x 
        
        
    def forward(self, x, E_true, z=None, mode='full'):
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z 
        elif mode == 'decode':
            return self.decode(torch.cat((z,E_true), 1)) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), mu, logvar, z
            
    
    

    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
##########################################################################################
#################################### Latent Gen ##########################################
##########################################################################################
    
    
class LatentGeneratorV1(nn.Module):
    def __init__(self, out_size=128, latent_size=512, ndf=256, bias=True):
        super(LatentGeneratorV1, self).__init__()    
        self.out_size = out_size
        self.latent_size = latent_size
        self.ndf = ndf

        self.fc1 = torch.nn.Linear(self.out_size,   self.ndf, bias=bias)
        self.dr1 = torch.nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.dr2 = torch.nn.Dropout(p=0.5)
        self.fc3 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.dr3 = torch.nn.Dropout(p=0.5)
        self.fc4 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.dr4 = torch.nn.Dropout(p=0.5)
        self.fc5 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.dr5 = torch.nn.Dropout(p=0.5)
        self.fc6 = torch.nn.Linear(self.ndf, self.out_size,   bias=bias)
 

    def forward(self, noise):                
        img = self.dr1(F.elu(self.fc1(noise)))
        img = self.dr2(F.elu(self.fc2(img)))
        img = self.dr3(F.elu(self.fc3(img)))
        img = self.dr4(F.elu(self.fc4(img)))
        img = self.dr5(F.elu(self.fc5(img)))
        img = self.fc6(img)
        
        x1 = img
        x2 = torch.randn(x1.size(0), int(self.latent_size-self.out_size), device=x1.get_device())
        z = torch.cat((x1, x2), 1)

        return z
    
    

class LatentGenDiscV1(nn.Module):
    def __init__(self, out_size=128, ndf=256, bias=True):
        super(LatentGenDiscV1, self).__init__()    
        self.out_size = out_size
        self.ndf = ndf

        self.fc1 = torch.nn.Linear(self.out_size,   self.ndf, bias=bias)
        self.dr1 = torch.nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.dr2 = torch.nn.Dropout(p=0.5)
        self.fc3 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.dr3 = torch.nn.Dropout(p=0.5)
        self.fc4 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.dr4 = torch.nn.Dropout(p=0.5)
        self.fc5 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.dr5 = torch.nn.Dropout(p=0.5)
        self.fc6 = torch.nn.Linear(self.ndf, 1,   bias=bias)
 

    def forward(self, noise):                
        img = self.dr1(F.leaky_relu(self.fc1(noise)))
        img = self.dr2(F.leaky_relu(self.fc2(img)))
        img = self.dr3(F.leaky_relu(self.fc3(img)))
        img = self.dr4(F.leaky_relu(self.fc4(img)))
        img = self.dr5(F.leaky_relu(self.fc5(img)))
        img = self.fc6(img)

        return img

    
    
    
    
class LatentGeneratorV2_GaussMix(nn.Module):
    def __init__(self, out_size=128, latent_size=512, ndf=512, bias=True):
        super(LatentGeneratorV2_GaussMix, self).__init__()    
        self.out_size = out_size
        self.latent_size = latent_size
        self.ndf = ndf
        self.z_rand = self.latent_size - self.out_size

        self.fc1 = torch.nn.Linear(self.out_size,   self.ndf, bias=bias)
        self.fc2 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc3 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc4 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc5 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc6a = torch.nn.Linear(self.ndf, self.ndf,   bias=bias)
        self.fc6b = torch.nn.Linear(self.ndf, self.ndf,   bias=bias)
        self.fc7a = torch.nn.Linear(self.ndf, self.out_size,   bias=bias)
        self.fc7b = torch.nn.Linear(self.ndf, self.out_size,   bias=bias)
 
    def encode(self, noise):                
        img = F.leaky_relu(self.fc1(noise))
        img = F.leaky_relu(self.fc2(img))
        img = F.leaky_relu(self.fc3(img))
        img = F.leaky_relu(self.fc4(img))
        img = F.leaky_relu(self.fc5(img))
        mu  = F.leaky_relu(self.fc6a(img))
        sig = F.leaky_relu(self.fc6b(img))
        mu  = (self.fc7a(mu))
        sig = (self.fc7b(sig))
        
        return torch.cat((mu,torch.zeros(mu.size(0), self.z_rand, device = mu.device)), 1), torch.cat((sig,torch.zeros(sig.size(0), self.z_rand, device = mu.device)), 1)    
    

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std    
    
    def forward(self, noise):                
        mu, logvar = self.encode(noise)
        z = self.reparameterize(mu, logvar)
        return z
    
    
    
    
    
class LatentGeneratorV2(nn.Module):
    def __init__(self, out_size=128, latent_size=512, ndf=512, bias=True):
        super(LatentGeneratorV2, self).__init__()    
        self.out_size = out_size
        self.latent_size = latent_size
        self.ndf = ndf
        self.z_rand = self.latent_size - self.out_size

        self.fc1 = torch.nn.Linear(self.out_size,   self.ndf, bias=bias)
        self.fc2 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc3 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc4 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc5 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc6 = torch.nn.Linear(self.ndf, self.ndf,   bias=bias)
        self.fc7 = torch.nn.Linear(self.ndf, self.out_size,   bias=bias)
 
    def encode(self, noise):                
        img = F.leaky_relu(self.fc1(noise))
        img = F.leaky_relu(self.fc2(img))
        img = F.leaky_relu(self.fc3(img))
        img = F.leaky_relu(self.fc4(img))
        img = F.leaky_relu(self.fc5(img))
        img = F.leaky_relu(self.fc6(img))
        img = (self.fc7(img))
        
        return img
    
    def forward(self, noise):       
        x1 = self.encode(noise)
        x2 = torch.randn(x1.size(0), int(self.latent_size-self.out_size), device=x1.get_device())
        z = torch.cat((x1, x2), 1)

        return z

    
    
    
    
        
class LatentGenDiscV2(nn.Module):
    def __init__(self, out_size=128, ndf=512, bias=True):
        super(LatentGenDiscV2, self).__init__()    
        self.out_size = out_size
        self.ndf = ndf

        self.fc1 = torch.nn.Linear(self.out_size,   self.ndf, bias=bias)
        self.fc2 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc3 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc4 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc5 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc6 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc7 = torch.nn.Linear(self.ndf, 1,   bias=bias)
 

    def forward(self, noise):                
        img = F.leaky_relu(self.fc1(noise))
        img = F.leaky_relu(self.fc2(img))
        img = F.leaky_relu(self.fc3(img))
        img = F.leaky_relu(self.fc4(img))
        img = F.leaky_relu(self.fc5(img))
        img = F.leaky_relu(self.fc6(img))
        img = self.fc7(img)

        return img

    
    
    
    

class LatentGeneratorV3(nn.Module):
    def __init__(self, out_size=128, latent_size=512, ndf=256, bias=True):
        super(LatentGeneratorV3, self).__init__()    
        self.out_size = out_size
        self.latent_size = latent_size
        self.ndf = ndf

        self.fc1 = torch.nn.Linear(self.out_size,   self.ndf, bias=bias)
        self.dr1 = torch.nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.dr2 = torch.nn.Dropout(p=0.5)
        self.fc3 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.dr3 = torch.nn.Dropout(p=0.5)
        self.fc4 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.dr4 = torch.nn.Dropout(p=0.5)
        self.fc5 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.dr5 = torch.nn.Dropout(p=0.5)
        self.fc6a = torch.nn.Linear(self.ndf, self.out_size,   bias=bias)
        self.fc6b = torch.nn.Linear(self.ndf, self.out_size,   bias=bias)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std   
 

    def forward(self, noise, mode='train'):                
        img = self.dr1(F.elu(self.fc1(noise)))
        img = self.dr2(F.elu(self.fc2(img)))
        img = self.dr3(F.elu(self.fc3(img)))
        img = self.dr4(F.elu(self.fc4(img)))
        img = self.dr5(F.elu(self.fc5(img)))
        mu = self.fc6a(img)
        logvar = self.fc6b(img)
        
        mu1 = mu
        mu2 = torch.zeros(mu1.size(0), int(self.latent_size-self.out_size), device=mu1.get_device())
        mu = torch.cat((mu1, mu2), 1)

        logvar1 = logvar
        logvar2 = torch.zeros(logvar1.size(0), int(self.latent_size-self.out_size), device=logvar1.get_device())
        logvar = torch.cat((logvar1, logvar2), 1)

        if mode == 'reparameterize':
            return self.reparameterize(mu, logvar)
        else:
            return mu, logvar
    
    

class LatentGenDiscV3(nn.Module):
    def __init__(self, out_size=128, ndf=256, bias=True):
        super(LatentGenDiscV3, self).__init__()    
        self.out_size = out_size
        self.ndf = ndf

        self.fc1 = torch.nn.Linear(self.out_size*2 ,   self.ndf, bias=bias)
        self.dr1 = torch.nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.dr2 = torch.nn.Dropout(p=0.5)
        self.fc3 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.dr3 = torch.nn.Dropout(p=0.5)
        self.fc4 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.dr4 = torch.nn.Dropout(p=0.5)
        self.fc5 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.dr5 = torch.nn.Dropout(p=0.5)
        self.fc6 = torch.nn.Linear(self.ndf, 1,   bias=bias)
 

    def forward(self, noise):                
        img = self.dr1(F.leaky_relu(self.fc1(noise)))
        img = self.dr2(F.leaky_relu(self.fc2(img)))
        img = self.dr3(F.leaky_relu(self.fc3(img)))
        img = self.dr4(F.leaky_relu(self.fc4(img)))
        img = self.dr5(F.leaky_relu(self.fc5(img)))
        img = self.fc6(img)

        return img

    
    
    
    
    
class LatentGeneratorV4(nn.Module):
    def __init__(self, out_size=128, latent_size=512, ndf=256, bias=True):
        super(LatentGeneratorV4, self).__init__()    
        self.out_size = out_size
        self.latent_size = latent_size
        self.ndf = ndf

        self.fc1 = torch.nn.Linear(self.out_size,   self.ndf, bias=bias)
        self.fc2 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc3 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc4 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc5 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc6 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc7a = torch.nn.Linear(self.ndf, self.out_size,   bias=bias)
        self.fc7b = torch.nn.Linear(self.ndf, self.out_size,   bias=bias)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std   
 

    def forward(self, noise, mode='train'):                
        img = F.elu(self.fc1(noise))
        img = F.elu(self.fc2(img))
        img = F.elu(self.fc3(img))
        img = F.elu(self.fc4(img))
        img = F.elu(self.fc5(img))
        img = F.elu(self.fc6(img))
        mu = self.fc7a(img)
        logvar = self.fc7b(img)
        
        mu1 = mu
        mu2 = torch.zeros(mu1.size(0), int(self.latent_size-self.out_size), device=mu1.get_device())
        mu = torch.cat((mu1, mu2), 1)

        logvar1 = logvar
        logvar2 = torch.zeros(logvar1.size(0), int(self.latent_size-self.out_size), device=logvar1.get_device())
        logvar = torch.cat((logvar1, logvar2), 1)

        if mode == 'reparameterize':
            return self.reparameterize(mu, logvar)
        else:
            return mu, logvar
    
    

class LatentGenDiscV4(nn.Module):
    def __init__(self, out_size=128, ndf=256, bias=True):
        super(LatentGenDiscV4, self).__init__()    
        self.out_size = out_size
        self.ndf = ndf

        self.fc1 = torch.nn.Linear(self.out_size*2 ,   self.ndf, bias=bias)
        self.fc2 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc3 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc4 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc5 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc6 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc7 = torch.nn.Linear(self.ndf, 1,   bias=bias)
 

    def forward(self, noise):                
        img = F.leaky_relu(self.fc1(noise))
        img = F.leaky_relu(self.fc2(img))
        img = F.leaky_relu(self.fc3(img))
        img = F.leaky_relu(self.fc4(img))
        img = F.leaky_relu(self.fc5(img))
        img = F.leaky_relu(self.fc6(img))
        img = self.fc7(img)

        return img

    
    
    
    
    
class LatentGeneratorV5(nn.Module):
    def __init__(self, out_size=128, latent_size=512, ndf=256, bias=True):
        super(LatentGeneratorV5, self).__init__()    
        self.out_size = out_size
        self.latent_size = latent_size
        self.ndf = ndf

        self.fc1 = torch.nn.Linear(self.out_size,   self.ndf, bias=bias)
        self.fc2 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc3 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc4a = torch.nn.Linear(self.ndf, self.out_size,   bias=bias)
        self.fc4b = torch.nn.Linear(self.ndf, self.out_size,   bias=bias)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std   
 

    def forward(self, noise, mode='train'):                
        img = F.elu(self.fc1(noise))
        img = F.elu(self.fc2(img))
        img = F.elu(self.fc3(img))
        mu     = self.fc4a(img)
        logvar = self.fc4b(img)
        
        mu1 = mu
        mu2 = torch.zeros(mu1.size(0), int(self.latent_size-self.out_size), device=mu1.get_device())
        mu = torch.cat((mu1, mu2), 1)

        logvar1 = logvar
        logvar2 = torch.zeros(logvar1.size(0), int(self.latent_size-self.out_size), device=logvar1.get_device())
        logvar = torch.cat((logvar1, logvar2), 1)

        if mode == 'reparameterize':
            return self.reparameterize(mu, logvar)
        else:
            return mu, logvar
    
    

class LatentGenDiscV5(nn.Module):
    def __init__(self, out_size=128, ndf=256, bias=True):
        super(LatentGenDiscV5, self).__init__()    
        self.out_size = out_size
        self.ndf = ndf

        self.fc1 = torch.nn.Linear(self.out_size*2 ,   self.ndf, bias=bias)
        self.fc2 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc3 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc4 = torch.nn.Linear(self.ndf, 1,   bias=bias)
 

    def forward(self, noise):                
        img = F.leaky_relu(self.fc1(noise))
        img = F.leaky_relu(self.fc2(img))
        img = F.leaky_relu(self.fc3(img))
        img = self.fc4(img)

        return img

    
    
    
    
    
    
    

    
    
    
    
    
class LatentGeneratorV6(nn.Module):
    def __init__(self, out_size=128, latent_size=512, ndf=256, bias=True):
        super(LatentGeneratorV6, self).__init__()    
        self.out_size = out_size
        self.latent_size = latent_size
        self.ndf = ndf

        self.fc1 = torch.nn.Linear(self.out_size,   self.ndf, bias=bias)
        self.fc2 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc3 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc4a = torch.nn.Linear(self.ndf, self.ndf,   bias=bias)
        self.fc4b = torch.nn.Linear(self.ndf, self.ndf,   bias=bias)
        self.fc5a = torch.nn.Linear(self.ndf, self.ndf,   bias=bias)
        self.fc5b = torch.nn.Linear(self.ndf, self.ndf,   bias=bias)
        self.fc6a = torch.nn.Linear(self.ndf, self.out_size,   bias=bias)
        self.fc6b = torch.nn.Linear(self.ndf, self.out_size,   bias=bias)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std   
 

    def forward(self, noise, mode='train'):                
        img = F.elu(self.fc1(noise))
        img = F.elu(self.fc2(img))
        img = F.elu(self.fc3(img))
        mu     = F.elu(self.fc4a(img))
        logvar = F.elu(self.fc4b(img))
        mu     = F.elu(self.fc5a(img))
        logvar = F.elu(self.fc5b(img))
        mu     = self.fc6a(img)
        logvar = self.fc6b(img)
        
        mu1 = mu
        mu2 = torch.zeros(mu1.size(0), int(self.latent_size-self.out_size), device=mu1.get_device())
        mu = torch.cat((mu1, mu2), 1)

        logvar1 = logvar
        logvar2 = torch.zeros(logvar1.size(0), int(self.latent_size-self.out_size), device=logvar1.get_device())
        logvar = torch.cat((logvar1, logvar2), 1)

        if mode == 'reparameterize':
            return self.reparameterize(mu, logvar)
        else:
            return mu, logvar
    
    

class LatentGenDiscV6(nn.Module):
    def __init__(self, out_size=128, ndf=256, bias=True):
        super(LatentGenDiscV6, self).__init__()    
        self.out_size = out_size
        self.ndf = ndf

        self.fc1 = torch.nn.Linear(self.out_size*2 ,   self.ndf, bias=bias)
        self.fc2 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc3 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc4 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc5 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc6 = torch.nn.Linear(self.ndf, 1,   bias=bias)
 

    def forward(self, noise):                
        img = F.leaky_relu(self.fc1(noise))
        img = F.leaky_relu(self.fc2(img))
        img = F.leaky_relu(self.fc3(img))
        img = F.leaky_relu(self.fc4(img))
        img = F.leaky_relu(self.fc5(img))
        img = self.fc6(img)

        return img

    
    
    
    

class Latent_Net_V1(nn.Module):
    def __init__(self, out_size=128, ndf=256, bias=True):
        super(Latent_Net_V1, self).__init__()    
        self.out_size = out_size
        self.ndf = ndf

        self.fc1 = torch.nn.Linear(self.out_size ,self.ndf, bias=bias)
        self.fc2 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc3 = torch.nn.Linear(self.ndf, self.out_size, bias=bias)


    def forward(self, noise):                
        img = F.leaky_relu(self.fc1(noise))
        img = F.leaky_relu(self.fc2(img))
        img = self.fc3(img)

        return img
    
    
    
    
    
    

class Latent_Net_V2(nn.Module):
    def __init__(self, out_size=128, ndf=256, bias=True):
        super(Latent_Net_V2, self).__init__()    
        self.out_size = out_size
        self.ndf = ndf

        self.fc1 = torch.nn.Linear(self.out_size ,self.ndf, bias=bias)
        self.fc2 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc3 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc4 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc5 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc6 = torch.nn.Linear(self.ndf, self.out_size, bias=bias)


    def forward(self, noise):                
        img = F.leaky_relu(self.fc1(noise))
        img = F.leaky_relu(self.fc2(img))
        img = F.leaky_relu(self.fc3(img))
        img = F.leaky_relu(self.fc4(img))
        img = self.fc5(img)
        img = self.fc6(img)

        return img    
    
    
    
    
class Latent_Net_V3(nn.Module):
    def __init__(self, out_size=128, ndf=256, bias=True):
        super(Latent_Net_V3, self).__init__()    
        self.out_size = out_size
        self.ndf = ndf

        self.fc1 = torch.nn.Linear(self.out_size ,self.ndf, bias=bias)
        self.fc2 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc3 = torch.nn.Linear(self.ndf, self.out_size, bias=bias)
        self.fc4 = torch.nn.Linear(self.out_size, self.ndf, bias=bias)
        self.fc5 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc6 = torch.nn.Linear(self.ndf, self.out_size, bias=bias)


    def forward(self, noise): 
        split1 = noise
        img = F.leaky_relu(self.fc1(noise))
        img = F.leaky_relu(self.fc2(img))
        img = self.fc3(img)
        
        img = split1 + img
        
        split2 = img
        img = F.leaky_relu(self.fc4(img))
        img = F.leaky_relu(self.fc5(img))
        img = self.fc6(img)

        img = split2 + img

        return img    
    
    
    
    
    
    
    
class Latent_Net_V4(nn.Module):
    def __init__(self, out_size=128, ndf=256, bias=True):
        super(Latent_Net_V4, self).__init__()    
        self.out_size = out_size
        self.ndf = ndf

        self.fc1 = torch.nn.Linear(self.out_size ,self.ndf, bias=bias)
        self.fc2 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc3 = torch.nn.Linear(self.ndf, self.out_size, bias=bias)
        self.fc4 = torch.nn.Linear(self.out_size, self.ndf, bias=bias)
        self.fc5 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc6 = torch.nn.Linear(self.ndf, self.out_size, bias=bias)


    def forward(self, noise): 
        split1 = noise
        img = F.leaky_relu(self.fc1(noise))
        img = F.leaky_relu(self.fc2(img))
        img = self.fc3(img)
        
        img = split1 + img
        
        split2 = img
        img = F.leaky_relu(self.fc4(img))
        img = F.leaky_relu(self.fc5(img))
        img = self.fc6(img)

        img = split2 + img

        return img        
    
   
    
    
    
    
    
    
class Latent_Net_V5(nn.Module):
    def __init__(self, out_size=128, ndf=256, z_enc=12, bias=True):
        super(Latent_Net_V5, self).__init__()    
        self.out_size = out_size
        self.z_enc = z_enc
        self.ndf = ndf

        self.fc1 = torch.nn.Linear(self.z_enc, self.ndf, bias=bias)
        self.fc2 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc3 = torch.nn.Linear(self.ndf, self.z_enc, bias=bias)
        self.fc4 = torch.nn.Linear(self.z_enc, self.ndf, bias=bias)
        self.fc5 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc6 = torch.nn.Linear(self.ndf, self.z_enc, bias=bias)


    def forward(self, noise): 
        rand = noise[:,self.z_enc:]
        enc =  noise[:,:self.z_enc]
        
        split1 = enc
        enc = F.leaky_relu(self.fc1(enc))
        enc = F.leaky_relu(self.fc2(enc))
        enc = self.fc3(enc)
        
        enc = split1 + enc
        
        split2 = enc
        enc = F.leaky_relu(self.fc4(enc))
        enc = F.leaky_relu(self.fc5(enc))
        enc = self.fc6(enc)

        enc = split2 + enc
        
        noise = torch.cat((enc, rand), 1)

        return noise    
    
    
    
    
    
    
    
    
class Latent_Net_V5_Energy(nn.Module):
    def __init__(self, out_size=128, ndf=256, z_enc=12, bias=True):
        super(Latent_Net_V5_Energy, self).__init__()    
        self.out_size = out_size
        self.z_enc = z_enc
        self.ndf = ndf

        self.fc1 = torch.nn.Linear(self.z_enc+1, self.ndf, bias=bias)
        self.fc2 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc3 = torch.nn.Linear(self.ndf, self.z_enc, bias=bias)
        self.fc4 = torch.nn.Linear(self.z_enc, self.ndf, bias=bias)
        self.fc5 = torch.nn.Linear(self.ndf, self.ndf, bias=bias)
        self.fc6 = torch.nn.Linear(self.ndf, self.z_enc, bias=bias)


    def forward(self, noise, energy): 
        rand = noise[:,self.z_enc:]
        enc =  noise[:,:self.z_enc]
        
        
        split1 = enc
        enc = torch.cat((enc, energy),1)

        enc = F.leaky_relu(self.fc1(enc))
        enc = F.leaky_relu(self.fc2(enc))
        enc = self.fc3(enc)
        
        enc = split1 + enc
        
        split2 = enc
        enc = F.leaky_relu(self.fc4(enc))
        enc = F.leaky_relu(self.fc5(enc))
        enc = self.fc6(enc)

        enc = split2 + enc
        
        noise = torch.cat((enc, rand), 1)

        return noise    
    
    
    
    
    
    
    
    
    
    
    
   
  
    
class Discriminator_GettingH(nn.Module):
    def __init__(self, isize=30, nc=2, ndf=128):
        super(Discriminator_GettingH, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc

        
        self.conv1b = torch.nn.Conv3d(1, ndf, kernel_size=3, stride=(2,1,1), padding=0, bias=False)
        self.bn1b = torch.nn.LayerNorm([23,11,11])
        self.conv2b = torch.nn.Conv3d(ndf, ndf, kernel_size=3, stride=(2,1,1), padding=0, bias=False)
        self.bn2b = torch.nn.LayerNorm([11,9,9])
        self.conv3b = torch.nn.Conv3d(ndf, ndf, kernel_size=3, stride=(1,1,1), padding=0, bias=False)


        self.conv1c = torch.nn.Conv3d(1, ndf, kernel_size=3, stride=(2,1,1), padding=0, bias=False)
        self.bn1c = torch.nn.LayerNorm([23,11,11])
        self.conv2c = torch.nn.Conv3d(ndf, ndf, kernel_size=3, stride=(2,1,1), padding=0, bias=False)
        self.bn2c = torch.nn.LayerNorm([11,9,9])
        self.conv3c = torch.nn.Conv3d(ndf, ndf, kernel_size=3, stride=(1,1,1), padding=0, bias=False)

 
        #self.fc = torch.nn.Linear(5 * 4 * 4, 1)
        self.fc1a = torch.nn.Linear(48*13*13, int(ndf/2)) 
        self.fc1b = torch.nn.Linear(ndf * 9 * 7 * 7, int(ndf/2))
        self.fc1c = torch.nn.Linear(ndf * 9 * 7 * 7, int(ndf/2))
        self.fc1e = torch.nn.Linear(1, int(ndf/2))
        self.fc2 = torch.nn.Linear(int(ndf/2)*4, ndf*2)
        self.fc3 = torch.nn.Linear(ndf*2, ndf*2)
        self.fc4 = torch.nn.Linear(ndf*2, ndf*2)
        self.fc5 = torch.nn.Linear(ndf*2, 1)


    def forward(self, img, E_true):
        imga = img[:,0:1,:,:,:] #data.view(-1,1,30,30)
        imgb = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)

        img = imgb - imga
        
        xb = F.leaky_relu(self.bn1b(self.conv1b(imgb)), 0.2)
        xb = F.leaky_relu(self.bn2b(self.conv2b(xb)), 0.2)
        xb = F.leaky_relu(self.conv3b(xb), 0.2)
        xb = xb.view(-1, self.ndf * 9 * 7 * 7)
        xb = F.leaky_relu(self.fc1b(xb), 0.2)        
        
        imgb_relu = F.relu(imgb)
        
        xc = F.leaky_relu(self.bn1c(self.conv1c(torch.log(imgb_relu+1.0))), 0.2)
        xc = F.leaky_relu(self.bn2c(self.conv2c(xc)), 0.2)
        xc = F.leaky_relu(self.conv3c(xc), 0.2)
        xc = xc.view(-1, self.ndf * 9 * 7 * 7)
        xc = F.leaky_relu(self.fc1c(xc), 0.2)
        
        xb = torch.cat((xb, xc, F.leaky_relu(self.fc1a(img.view(-1, 48*13*13))) , F.leaky_relu(self.fc1e(E_true), 0.2)), 1)

        xb = F.leaky_relu(self.fc2(xb), 0.2)
        xb = F.leaky_relu(self.fc3(xb), 0.2)
        xb = F.leaky_relu(self.fc4(xb), 0.2)
        xb = self.fc5(xb)

        return xb.view(-1) ### flattens

    
    
    
    
    
    
class Latent_Critic_GettingH(nn.Module):
    def __init__(self, ):
        super(Latent_Critic_GettingH, self).__init__()
        self.linear1 = nn.Linear(1, 50)
        self.linear2 = nn.Linear(50, 100)        
        self.linear3 = nn.Linear(100, 50)
        self.linear4 = nn.Linear(50, 1)

    def forward(self, x):      
        x = F.leaky_relu(self.linear1(x.view(-1,1)), inplace=True)
        x = F.leaky_relu(self.linear2(x), inplace=True)
        x = F.leaky_relu(self.linear3(x), inplace=True)
        return self.linear4(x)
    

    
    
    

    
    
class BiBAE_F_3D_GettingH(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, device, nc=1, ngf=8, z_rand=500, z_enc=12):
        super(BiBAE_F_3D_GettingH, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        #self.z = z
        self.z_rand = z_rand
        self.z_enc = z_enc
        self.z_full = z_enc + z_rand
        self.args = args
        self.device = device

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(4,3,3), stride=(2,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.LayerNorm([24,13,13])
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(4,3,3), stride=(2,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.LayerNorm([12,13,13])
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,3,3), stride=(2,2,2),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.LayerNorm([6,7,7])
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=(3,3,3), stride=(1,1,1),
                               padding=(0,0,0), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.LayerNorm([4,5,5])

     
        self.fc1 = nn.Linear(4*5*5*ngf*8+1, ngf*500, bias=True)
        self.fc2 = nn.Linear(ngf*500, int(self.z_full*1.5), bias=True)
        
        self.fc31 = nn.Linear(int(self.z_full*1.5), self.z_enc, bias=True)
        self.fc32 = nn.Linear(int(self.z_full*1.5), self.z_enc, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z_full+1, int(self.z_full*1.5), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z_full*1.5), ngf*500, bias=True)
        self.cond3 = torch.nn.Linear(ngf*500, 12*7*7*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.LayerNorm([24,14,14])
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf*2, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.LayerNorm([48,28,28])

        #self.deconv3 = torch.nn.ConvTranspose3d(ngf*4, ngf*8, kernel_size=(3,3,3), stride=(2,2,2), padding=(0,1,1), bias=False)
        
        self.conv0 = torch.nn.Conv3d(ngf*2, ngf, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,0,0), bias=False)
        self.bnco0 = torch.nn.LayerNorm([48,13,13])
        self.conv1 = torch.nn.Conv3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco1 = torch.nn.LayerNorm([48,13,13])
        self.conv2 = torch.nn.Conv3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco2 = torch.nn.LayerNorm([48,13,13])
        self.conv3 = torch.nn.Conv3d(ngf*4, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.bnco3 = torch.nn.LayerNorm([48,13,13])
        self.conv4 = torch.nn.Conv3d(ngf*2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
    
        #self.dr03 = nn.Dropout(p=0.3, inplace=False)
        #self.dr05 = nn.Dropout(p=0.5, inplace=False)

    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,48,13,13))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        return torch.cat((self.fc31(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1), torch.cat((self.fc32(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        #print(std)
        #print(mu)
        return mu + eps*std

    def decode(self, z):
        ### need to do generated 30 layers, hence the loop!
        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond1(z), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond2(x), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond3(x), 0.2, inplace=True)

        ## change size for deconv2d network. Image is 10x10
        x = x.view(-1,self.ngf,12,7,7)        

        ## apply series of deconv2d and batch-norm
        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 24, 14, 14])), 0.2, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 48, 28, 28])), 0.2, inplace=True) #
        #x = F.leaky_relu(self.deconv3(x, output_size=[x.size(0), 1, 15, 120, 120]), 0.2, inplace=True) #

        ##Image is 120x120
        x = F.leaky_relu(self.bnco0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco3(self.conv3(x)), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        return x 
        
        
    def forward(self, x, E_true, z=None, mode='full'):
        #print(x.size())
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z 
        elif mode == 'decode':
            return self.decode(torch.cat((z,E_true), 1)) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), mu, logvar, z

        
        
        
        
        
        
        
class PostProcess_GettingH(nn.Module):
    def __init__(self, isize=30, nc=2, ndf=128, bias=False, out_funct='relu'):
        super(PostProcess_GettingH, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc
        self.bais = bias
        self.out_funct = out_funct
        
        self.fcec1 = torch.nn.Linear(2, int(ndf/2), bias=True)
        self.fcec2 = torch.nn.Linear(int(ndf/2), int(ndf/2), bias=True)
        self.fcec3 = torch.nn.Linear(int(ndf/2), int(ndf/2), bias=True)
        
        self.conv1 = torch.nn.Conv3d(1, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco1 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])
        
        self.conv2 = torch.nn.Conv3d(ndf+int(ndf/2), ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco2 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])

        self.conv3 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco3 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])

        self.conv4 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco4 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])

        self.conv5 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)

        self.conv6 = torch.nn.Conv3d(ndf, 1, kernel_size=1, stride=1, padding=0, bias=False)
 

    def forward(self, img, E_True=0):
        img = img.view(-1, 1, self.isize, self.isize, self.isize)
                
        econd = torch.cat((torch.sum(img.view(-1, self.isize*self.isize*self.isize), 1).view(-1, 1), E_True), 1)

        econd = F.leaky_relu(self.fcec1(econd), 0.01)
        econd = F.leaky_relu(self.fcec2(econd), 0.01)
        econd = F.leaky_relu(self.fcec3(econd), 0.01)
        
        econd = econd.view(-1, int(self.ndf/2), 1, 1, 1)
        econd = econd.expand(-1, -1, self.isize, self.isize, self.isize)        
        
        img = F.leaky_relu(self.bnco1(self.conv1(img)), 0.01)
        img = torch.cat((img, econd), 1)
        img = F.leaky_relu(self.bnco2(self.conv2(img)), 0.01)
        img = F.leaky_relu(self.bnco3(self.conv3(img)), 0.01)
        img = F.leaky_relu(self.bnco4(self.conv4(img)), 0.01)
        img = F.leaky_relu(self.conv5(img), 0.01) 
        img = self.conv6(img)

        if self.out_funct == 'relu':
            img = F.relu(img)
        elif self.out_funct == 'leaky_relu':
            img = F.leaky_relu(img, 0.01) 
              
        return img.view(-1, 1, self.isize, self.isize, self.isize)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class Discriminator_F_BatchStatV2_CoreAnat(nn.Module):
    def __init__(self, isize=48, nc=2, ndf=64):
        super(Discriminator_F_BatchStatV2_CoreAnat, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc
        self.size_embed = 16
        self.conv1_bias = False

        self.Embed1a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed1b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed1c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed2a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed2b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed2c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed3a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed3b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed3c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed4a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed4b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed4c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        
        self.embed_lin = torch.nn.Linear(self.size_embed*4, 64) 
        self.embed_log = torch.nn.Linear(self.size_embed*4, 64) 
        self.diff_lin = torch.nn.Linear(48*13*13, 64) 
        self.diff_log = torch.nn.Linear(48*13*13, 64) 
        
        
        self.conv1a = torch.nn.Conv3d(1, ndf, kernel_size=2, stride=(2,1,1), padding=0, bias=False)
        self.ln1a = torch.nn.LayerNorm([24,12,12])
        self.conv1b = torch.nn.Conv3d(ndf, ndf, kernel_size=2, stride=(2,1,1), padding=0, bias=False)
        self.ln1b = torch.nn.LayerNorm([12,11,11])
        self.conv1c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=(2,2,2), padding=(1,2,2), bias=False)
        
        self.conv2a = torch.nn.Conv3d(1, ndf, kernel_size=2, stride=(2,1,1), padding=0, bias=False)
        self.ln2a = torch.nn.LayerNorm([24,12,12])
        self.conv2b = torch.nn.Conv3d(ndf, ndf, kernel_size=2, stride=(2,1,1), padding=0, bias=False)
        self.ln2b = torch.nn.LayerNorm([12,11,11])
        self.conv2c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=(2,2,2), padding=(1,2,2), bias=False)

        self.conv_lin = torch.nn.Linear(6*6*6*ndf, 64) 
        self.conv_log = torch.nn.Linear(6*6*6*ndf, 64) 
        
        self.econd_lin = torch.nn.Linear(1, 64) 

        self.fc1 = torch.nn.Linear(64*7, 256)
        self.fc2 = torch.nn.Linear(256,  256)
        self.fc3 = torch.nn.Linear(256, 1)


    def forward(self, img, E_true):
        batch_size = img.size(0)
        
        #img_log_a = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)
        #img_log_b = img[:,0:1,:,:,:] #data.view(-1,1,30,30)

        #print('img_log_a', torch.sum(img_log_a))
        #print('img_log_b', torch.sum(img_log_b))

        #img_a = torch.exp(img_log_a-5.0)-np.exp(-5.0)
        #img_b = torch.exp(img_log_b-5.0)-np.exp(-5.0)
        
        img_a = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)
        img_b = img[:,0:1,:,:,:] #data.view(-1,1,30,30)
        
        img_log_a = torch.log(img_a+np.exp(-5.0))+5.0
        img_log_b = torch.log(img_b+np.exp(-5.0))+5.0
        
        
        img_a_hit_std = torch.std(img_a.view(batch_size, -1), 1)
        img_a_hit_std_a = img_a_hit_std.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_a_hit_std_b = img_a_hit_std.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_a_hit_std = img_a_hit_std_a - img_a_hit_std_b
        
        img_a_hit_sum = torch.sum(img_a.view(batch_size, -1), 1)
        img_a_hit_sum_a = img_a_hit_sum.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_a_hit_sum_b = img_a_hit_sum.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_a_hit_sum = img_a_hit_sum_a - img_a_hit_sum_b

        
        img_a_hit_std = torch.transpose(img_a_hit_std, 1, 2)
        img_a_hit_sum = torch.transpose(img_a_hit_sum, 1, 2)

        
        img_a_hit_std = F.leaky_relu(self.Embed1a(img_a_hit_std))
        img_a_hit_std = F.leaky_relu(self.Embed1b(img_a_hit_std))
        img_a_hit_std = F.leaky_relu(self.Embed1c(img_a_hit_std))
        
        img_a_hit_sum = F.leaky_relu(self.Embed2a(img_a_hit_sum))
        img_a_hit_sum = F.leaky_relu(self.Embed2b(img_a_hit_sum))
        img_a_hit_sum = F.leaky_relu(self.Embed2c(img_a_hit_sum))

        img_a_hit_std_std  =  torch.std(img_a_hit_std, 2)
        img_a_hit_std_mean = torch.mean(img_a_hit_std, 2)

        img_a_hit_sum_std  =  torch.std(img_a_hit_sum, 2)
        img_a_hit_sum_mean = torch.mean(img_a_hit_sum, 2)
        
        xa1 = torch.cat((img_a_hit_std_std, img_a_hit_std_mean, img_a_hit_sum_std, img_a_hit_sum_mean), 1)
        xa1 = F.leaky_relu(self.embed_lin(xa1), 0.2)
        
        
        img_log_a_hit_std = torch.std(img_log_a.view(batch_size, -1), 1)
        img_log_a_hit_std_a = img_log_a_hit_std.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_log_a_hit_std_b = img_log_a_hit_std.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_log_a_hit_std = img_log_a_hit_std_a - img_log_a_hit_std_b
        
        img_log_a_hit_sum = torch.sum(img_log_a.view(batch_size, -1), 1)
        img_log_a_hit_sum_a = img_log_a_hit_sum.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_log_a_hit_sum_b = img_log_a_hit_sum.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_log_a_hit_sum = img_log_a_hit_sum_a - img_log_a_hit_sum_b
        
        img_log_a_hit_std = torch.transpose(img_log_a_hit_std, 1, 2)
        img_log_a_hit_sum = torch.transpose(img_log_a_hit_sum, 1, 2)
        
        img_log_a_hit_std = F.leaky_relu(self.Embed1a(img_log_a_hit_std))
        img_log_a_hit_std = F.leaky_relu(self.Embed1b(img_log_a_hit_std))
        img_log_a_hit_std = F.leaky_relu(self.Embed1c(img_log_a_hit_std))
        
        img_log_a_hit_sum = F.leaky_relu(self.Embed2a(img_log_a_hit_sum))
        img_log_a_hit_sum = F.leaky_relu(self.Embed2b(img_log_a_hit_sum))
        img_log_a_hit_sum = F.leaky_relu(self.Embed2c(img_log_a_hit_sum))

        img_log_a_hit_std_std  =  torch.std(img_log_a_hit_std, 2)
        img_log_a_hit_std_mean = torch.mean(img_log_a_hit_std, 2)

        img_log_a_hit_sum_std  =  torch.std(img_log_a_hit_sum, 2)
        img_log_a_hit_sum_mean = torch.mean(img_log_a_hit_sum, 2)        
        
        xa2 = torch.cat((img_log_a_hit_std_std, img_log_a_hit_std_mean, img_log_a_hit_sum_std, img_log_a_hit_sum_mean), 1)
        xa2 = F.leaky_relu(self.embed_lin(xa2), 0.2)
        
               
        xa3 = F.leaky_relu(self.diff_lin((img_a-img_b).view(batch_size, -1)), 0.2)
        
        
        xa4 = F.leaky_relu(self.diff_log((img_log_a-img_log_b).view(batch_size, -1)), 0.2)
 

        xa5 = F.leaky_relu(self.ln1a(self.conv1a(img_a*0.001)), 0.2)
        xa5 = F.leaky_relu(self.ln1b(self.conv1b(xa5)), 0.2)
        xa5 = F.leaky_relu(self.conv1c(xa5), 0.2)
        xa5 = xa5.view(-1, self.ndf*6*6*6)
        xa5 = F.leaky_relu(self.conv_lin(xa5), 0.2)
        

        xa6 = F.leaky_relu(self.ln2a(self.conv2a(img_log_a)), 0.2)
        xa6 = F.leaky_relu(self.ln2b(self.conv2b(xa6)), 0.2)
        xa6 = F.leaky_relu(self.conv2c(xa6), 0.2)
        xa6 = xa6.view(-1, self.ndf*6*6*6)
        xa6 = F.leaky_relu(self.conv_log(xa6), 0.2)        
        
        xa7 = F.leaky_relu(self.econd_lin(E_true), 0.2)      
        
        xa = torch.cat((xa1, xa2, xa3, xa4, xa5, xa6, xa7), 1)
        
        #print('xa1', torch.sum(xa1))
        #print('xa2', torch.sum(xa2))
        #print('xa3', torch.sum(xa3))
        #print('xa4', torch.sum(xa4))
        #print('xa5', torch.sum(xa5))
        #print('xa6', torch.sum(xa6))
        #print('xa7', torch.sum(xa7))
        
        xa = F.leaky_relu(self.fc1(xa), 0.2)
        xa = F.leaky_relu(self.fc2(xa), 0.2)
        xa = self.fc3(xa)

        #print('xa', torch.sum(xa))
        
        return xa ### flattens
    
    
    

    
    
   
    
class Discriminator_F_BatchStat_SmlReco_CoreAnat(nn.Module):
    def __init__(self, isize=48, nc=2, ndf=64):
        super(Discriminator_F_BatchStat_SmlReco_CoreAnat, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc
        self.size_embed = 16
        self.conv1_bias = False

        self.Embed1a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed1b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed1c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed2a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed2b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed2c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed3a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed3b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed3c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed4a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed4b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed4c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        
        self.embed_lin = torch.nn.Linear(self.size_embed*4, 64) 
        self.embed_log = torch.nn.Linear(self.size_embed*4, 64) 
        self.diff_lin = torch.nn.Linear(48*5*5, 64) 
        self.diff_log = torch.nn.Linear(48*5*5, 64) 
        
        
        self.conv1a = torch.nn.Conv3d(1, ndf, kernel_size=2, stride=(2,1,1), padding=0, bias=False)
        self.ln1a = torch.nn.LayerNorm([24,12,12])
        self.conv1b = torch.nn.Conv3d(ndf, ndf, kernel_size=2, stride=(2,1,1), padding=0, bias=False)
        self.ln1b = torch.nn.LayerNorm([12,11,11])
        self.conv1c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=(2,2,2), padding=(1,2,2), bias=False)
        
        self.conv2a = torch.nn.Conv3d(1, ndf, kernel_size=2, stride=(2,1,1), padding=0, bias=False)
        self.ln2a = torch.nn.LayerNorm([24,12,12])
        self.conv2b = torch.nn.Conv3d(ndf, ndf, kernel_size=2, stride=(2,1,1), padding=0, bias=False)
        self.ln2b = torch.nn.LayerNorm([12,11,11])
        self.conv2c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=(2,2,2), padding=(1,2,2), bias=False)

        self.conv_lin = torch.nn.Linear(6*6*6*ndf, 64) 
        self.conv_log = torch.nn.Linear(6*6*6*ndf, 64) 
        
        self.econd_lin = torch.nn.Linear(1, 64) 

        self.fc1 = torch.nn.Linear(64*7, 256)
        self.fc2 = torch.nn.Linear(256,  256)
        self.fc3 = torch.nn.Linear(256, 1)


    def forward(self, img, E_true):
        batch_size = img.size(0)
        
        #img_log_a = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)
        #img_log_b = img[:,0:1,:,:,:] #data.view(-1,1,30,30)

        #print('img_log_a', torch.sum(img_log_a))
        #print('img_log_b', torch.sum(img_log_b))

        #img_a = torch.exp(img_log_a-5.0)-np.exp(-5.0)
        #img_b = torch.exp(img_log_b-5.0)-np.exp(-5.0)
        
        img_a = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)
        img_b = img[:,0:1,:,:,:] #data.view(-1,1,30,30)
        
        img_log_a = torch.log(img_a+np.exp(-5.0))+5.0
        img_log_b = torch.log(img_b+np.exp(-5.0))+5.0
        
        
        img_a_hit_std = torch.std(img_a.view(batch_size, -1), 1)
        img_a_hit_std_a = img_a_hit_std.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_a_hit_std_b = img_a_hit_std.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_a_hit_std = img_a_hit_std_a - img_a_hit_std_b
        
        img_a_hit_sum = torch.sum(img_a.view(batch_size, -1), 1)
        img_a_hit_sum_a = img_a_hit_sum.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_a_hit_sum_b = img_a_hit_sum.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_a_hit_sum = img_a_hit_sum_a - img_a_hit_sum_b

        
        img_a_hit_std = torch.transpose(img_a_hit_std, 1, 2)
        img_a_hit_sum = torch.transpose(img_a_hit_sum, 1, 2)

        
        img_a_hit_std = F.leaky_relu(self.Embed1a(img_a_hit_std))
        img_a_hit_std = F.leaky_relu(self.Embed1b(img_a_hit_std))
        img_a_hit_std = F.leaky_relu(self.Embed1c(img_a_hit_std))
        
        img_a_hit_sum = F.leaky_relu(self.Embed2a(img_a_hit_sum))
        img_a_hit_sum = F.leaky_relu(self.Embed2b(img_a_hit_sum))
        img_a_hit_sum = F.leaky_relu(self.Embed2c(img_a_hit_sum))

        img_a_hit_std_std  =  torch.std(img_a_hit_std, 2)
        img_a_hit_std_mean = torch.mean(img_a_hit_std, 2)

        img_a_hit_sum_std  =  torch.std(img_a_hit_sum, 2)
        img_a_hit_sum_mean = torch.mean(img_a_hit_sum, 2)
        
        xa1 = torch.cat((img_a_hit_std_std, img_a_hit_std_mean, img_a_hit_sum_std, img_a_hit_sum_mean), 1)
        xa1 = F.leaky_relu(self.embed_lin(xa1), 0.2)
        
        
        img_log_a_hit_std = torch.std(img_log_a.view(batch_size, -1), 1)
        img_log_a_hit_std_a = img_log_a_hit_std.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_log_a_hit_std_b = img_log_a_hit_std.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_log_a_hit_std = img_log_a_hit_std_a - img_log_a_hit_std_b
        
        img_log_a_hit_sum = torch.sum(img_log_a.view(batch_size, -1), 1)
        img_log_a_hit_sum_a = img_log_a_hit_sum.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_log_a_hit_sum_b = img_log_a_hit_sum.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_log_a_hit_sum = img_log_a_hit_sum_a - img_log_a_hit_sum_b
        
        img_log_a_hit_std = torch.transpose(img_log_a_hit_std, 1, 2)
        img_log_a_hit_sum = torch.transpose(img_log_a_hit_sum, 1, 2)
        
        img_log_a_hit_std = F.leaky_relu(self.Embed1a(img_log_a_hit_std))
        img_log_a_hit_std = F.leaky_relu(self.Embed1b(img_log_a_hit_std))
        img_log_a_hit_std = F.leaky_relu(self.Embed1c(img_log_a_hit_std))
        
        img_log_a_hit_sum = F.leaky_relu(self.Embed2a(img_log_a_hit_sum))
        img_log_a_hit_sum = F.leaky_relu(self.Embed2b(img_log_a_hit_sum))
        img_log_a_hit_sum = F.leaky_relu(self.Embed2c(img_log_a_hit_sum))

        img_log_a_hit_std_std  =  torch.std(img_log_a_hit_std, 2)
        img_log_a_hit_std_mean = torch.mean(img_log_a_hit_std, 2)

        img_log_a_hit_sum_std  =  torch.std(img_log_a_hit_sum, 2)
        img_log_a_hit_sum_mean = torch.mean(img_log_a_hit_sum, 2)        
        
        xa2 = torch.cat((img_log_a_hit_std_std, img_log_a_hit_std_mean, img_log_a_hit_sum_std, img_log_a_hit_sum_mean), 1)
        xa2 = F.leaky_relu(self.embed_lin(xa2), 0.2)
        
               
        xa3 = F.leaky_relu(self.diff_lin((img_a[:,:,:,4:9,4:9]-img_b[:,:,:,4:9,4:9]).view(batch_size, -1)), 0.2)
        
        
        xa4 = F.leaky_relu(self.diff_log((img_log_a[:,:,:,4:9,4:9]-img_log_b[:,:,:,4:9,4:9]).view(batch_size, -1)), 0.2)
 

        xa5 = F.leaky_relu(self.ln1a(self.conv1a(img_a*0.001)), 0.2)
        xa5 = F.leaky_relu(self.ln1b(self.conv1b(xa5)), 0.2)
        xa5 = F.leaky_relu(self.conv1c(xa5), 0.2)
        xa5 = xa5.view(-1, self.ndf*6*6*6)
        xa5 = F.leaky_relu(self.conv_lin(xa5), 0.2)
        

        xa6 = F.leaky_relu(self.ln2a(self.conv2a(img_log_a)), 0.2)
        xa6 = F.leaky_relu(self.ln2b(self.conv2b(xa6)), 0.2)
        xa6 = F.leaky_relu(self.conv2c(xa6), 0.2)
        xa6 = xa6.view(-1, self.ndf*6*6*6)
        xa6 = F.leaky_relu(self.conv_log(xa6), 0.2)        
        
        xa7 = F.leaky_relu(self.econd_lin(E_true), 0.2)      
        
        xa = torch.cat((xa1, xa2, xa3, xa4, xa5, xa6, xa7), 1)
        
        #print('xa1', torch.sum(xa1))
        #print('xa2', torch.sum(xa2))
        #print('xa3', torch.sum(xa3))
        #print('xa4', torch.sum(xa4))
        #print('xa5', torch.sum(xa5))
        #print('xa6', torch.sum(xa6))
        #print('xa7', torch.sum(xa7))
        
        xa = F.leaky_relu(self.fc1(xa), 0.2)
        xa = F.leaky_relu(self.fc2(xa), 0.2)
        xa = self.fc3(xa)

        #print('xa', torch.sum(xa))
        
        return xa ### flattens
    
    
    

    
    
    
    
    
    
    
    
class BiBAE_F_3D_BatchStat_CoreAnat(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, device, nc=1, ngf=8, z_rand=500, z_enc=12):
        super(BiBAE_F_3D_BatchStat_CoreAnat, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        self.z_rand = z_rand
        self.z_enc = z_enc
        self.z_full = z_enc + z_rand
        self.args = args
        self.device = device

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(2,2,2), stride=(2,1,1),
                               padding=(0,0,0), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.LayerNorm([24,12,12])
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(2,2,2), stride=(2,1,1),
                               padding=(0,0,0), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.LayerNorm([12,11,11])
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(1,2,2), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.LayerNorm([6,6,6])
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*4, kernel_size=(4,4,4), stride=(1,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.LayerNorm([5,5,5])

     
        self.fc1 = nn.Linear(5*5*5*ngf*4+1, ngf*100, bias=True)
        self.fc2 = nn.Linear(ngf*100, int(self.z_full), bias=True)
        
        self.fc31 = nn.Linear(int(self.z_full), self.z_enc, bias=True)
        self.fc32 = nn.Linear(int(self.z_full), self.z_enc, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z_full+1, int(self.z_full), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z_full), ngf*100, bias=True)
        self.cond3 = torch.nn.Linear(ngf*100, 7*7*12*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.LayerNorm([24,14,14])
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.LayerNorm([48,28,28])

        self.conv0a = torch.nn.Conv3d(ngf, ngf, kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0), bias=False)
        self.bnco0a = torch.nn.LayerNorm([48,14,14])
        self.conv0 = torch.nn.Conv3d(ngf, ngf, kernel_size=(3,4,4), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco0 = torch.nn.LayerNorm([48,13,13])
        self.conv1 = torch.nn.Conv3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco1 = torch.nn.LayerNorm([48,13,13])
        self.conv2 = torch.nn.Conv3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco2 = torch.nn.LayerNorm([48,13,13])
        self.conv3 = torch.nn.Conv3d(ngf*4, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.bnco3 = torch.nn.LayerNorm([48,13,13])
        self.conv4 = torch.nn.Conv3d(ngf*2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)


    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,48,13,13))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        return torch.cat((self.fc31(x),torch.zeros(x.size(0), self.z_rand, device = x.device)), 1), torch.cat((self.fc32(x),torch.zeros(x.size(0), self.z_rand, device = x.device)), 1)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)

        x = x.view(-1,self.ngf,12,7,7)        

        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 24, 14, 14])), 0.2, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 48, 28, 28])), 0.2, inplace=True) #


        x = F.leaky_relu(self.bnco0a(self.conv0a(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco3(self.conv3(x)), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        return x 
        
        
    def forward(self, x, E_true, z=None, mode='full'):
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z 
        elif mode == 'decode':
            return self.decode(torch.cat((z,E_true), 1)) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), mu, logvar, z
        elif mode == 'half':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1))
            
    
    

    
    
    
    
    
class PostProcess_Size1Conv_EcondV2_CoreAnat(nn.Module):
    def __init__(self, isize=48, isize2=13, nc=2, ndf=64, bias=False, out_funct='relu'):
        super(PostProcess_Size1Conv_EcondV2_CoreAnat, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.isize2 = isize2
        self.nc = nc
        self.bais = bias
        self.out_funct = out_funct
        
        self.fcec1 = torch.nn.Linear(2, int(ndf/2), bias=True)
        self.fcec2 = torch.nn.Linear(int(ndf/2), int(ndf/2), bias=True)
        self.fcec3 = torch.nn.Linear(int(ndf/2), int(ndf/2), bias=True)
        
        self.conv1 = torch.nn.Conv3d(1, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco1 = torch.nn.LayerNorm([self.isize, self.isize2, self.isize2])
        
        self.conv2 = torch.nn.Conv3d(ndf+int(ndf/2), ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco2 = torch.nn.LayerNorm([self.isize, self.isize2, self.isize2])

        self.conv3 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco3 = torch.nn.LayerNorm([self.isize, self.isize2, self.isize2])

        self.conv4 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco4 = torch.nn.LayerNorm([self.isize, self.isize2, self.isize2])

        self.conv5 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)

        self.conv6 = torch.nn.Conv3d(ndf, 1, kernel_size=1, stride=1, padding=0, bias=False)
 

    def forward(self, img, E_True=0):
        img = img.view(-1, 1, self.isize, self.isize2, self.isize2) 
                
        econd = torch.cat((torch.sum(img.view(-1, self.isize2*self.isize2*self.isize), 1).view(-1, 1), E_True), 1)

        econd = F.leaky_relu(self.fcec1(econd), 0.01)
        econd = F.leaky_relu(self.fcec2(econd), 0.01)
        econd = F.leaky_relu(self.fcec3(econd), 0.01)
        
        econd = econd.view(-1, int(self.ndf/2), 1, 1, 1)
        econd = econd.expand(-1, -1, self.isize, self.isize2, self.isize2)        
        
        img = F.leaky_relu(self.bnco1(self.conv1(img)), 0.01)
        img = torch.cat((img, econd), 1)
        img = F.leaky_relu(self.bnco2(self.conv2(img)), 0.01)
        img = F.leaky_relu(self.bnco3(self.conv3(img)), 0.01)
        img = F.leaky_relu(self.bnco4(self.conv4(img)), 0.01)
        img = F.leaky_relu(self.conv5(img), 0.01) 
        img = self.conv6(img)

        if self.out_funct == 'relu':
            img = F.relu(img)
        elif self.out_funct == 'leaky_relu':
            img = F.leaky_relu(img, 0.01) 
              
        return img.view(-1, 1, self.isize, self.isize2, self.isize2)


     

class PostProcess_Lin_CoreAnat(nn.Module):
    def __init__(self, isize=48, isize2=13, nc=2, ndf=64, bias=False, out_funct='relu'):
        super(PostProcess_Lin_CoreAnat, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.isize2 = isize2
        self.nc = nc
        self.bais = bias
        self.out_funct = out_funct
        
        self.fc1 = torch.nn.Linear(1,   ndf, bias=bias)
        self.fc2 = torch.nn.Linear(ndf, ndf, bias=bias)
        self.fc3 = torch.nn.Linear(ndf, ndf, bias=bias)
        self.fc4 = torch.nn.Linear(ndf, ndf, bias=bias)
        self.fc5 = torch.nn.Linear(ndf, ndf, bias=bias)
        self.fc6 = torch.nn.Linear(ndf, 1,   bias=False)

 

    def forward(self, img, E_True=0):
        img = img.view(-1, self.isize, self.isize2, self.isize2, 1)
                
        img = F.leaky_relu(self.fc1(img), 0.01)
        img = F.leaky_relu(self.fc2(img), 0.01)
        img = F.leaky_relu(self.fc3(img), 0.01)
        img = F.leaky_relu(self.fc4(img), 0.01)
        img = F.leaky_relu(self.fc5(img), 0.01) 
        img = self.fc6(img)

        if self.out_funct == 'relu':
            img = F.relu(img)
        elif self.out_funct == 'leaky_relu':
            img = F.leaky_relu(img, 0.01) 
              
        return img.view(-1, 1, self.isize, self.isize2, self.isize2)
    
    
    
    
    
class PostProcess_LinEcond_CoreAnat(nn.Module):
    def __init__(self, isize=48, isize2=13, nc=2, ndf=64, bias=False, out_funct='relu'):
        super(PostProcess_LinEcond_CoreAnat, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.isize2 = isize2
        self.nc = nc
        self.bais = bias
        self.out_funct = out_funct
        
        self.fc1 = torch.nn.Linear(2,   ndf, bias=bias)
        self.fc2 = torch.nn.Linear(ndf, ndf, bias=bias)
        self.fc3 = torch.nn.Linear(ndf, ndf, bias=bias)
        self.fc4 = torch.nn.Linear(ndf, ndf, bias=bias)
        self.fc5 = torch.nn.Linear(ndf, ndf, bias=bias)
        self.fc6 = torch.nn.Linear(ndf, 1,   bias=False)

 

    def forward(self, img, E_True=0):
        img = img.view(-1, self.isize, self.isize2, self.isize2, 1)
        E_True = E_True.view(-1, 1, 1, 1, 1).repeat(1, self.isize, self.isize2, self.isize2, 1)
                
        img = torch.cat((img, E_True), 4)
            
        img = F.leaky_relu(self.fc1(img), 0.01)
        img = F.leaky_relu(self.fc2(img), 0.01)
        img = F.leaky_relu(self.fc3(img), 0.01)
        img = F.leaky_relu(self.fc4(img), 0.01)
        img = F.leaky_relu(self.fc5(img), 0.01) 
        img = self.fc6(img)

        if self.out_funct == 'relu':
            img = F.relu(img)
        elif self.out_funct == 'leaky_relu':
            img = F.leaky_relu(img, 0.01) 
              
        return img.view(-1, 1, self.isize, self.isize2, self.isize2)
    
    
    
    
    

    
    
    
    
    
    
    
    
   
    
class Discriminator_F_BatchStat_SmlReco_Core25(nn.Module):
    def __init__(self, isize=48, nc=2, ndf=64):
        super(Discriminator_F_BatchStat_SmlReco_Core25, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc
        self.size_embed = 16
        self.conv1_bias = False

        self.Embed1a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed1b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed1c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed2a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed2b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed2c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed3a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed3b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed3c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed4a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed4b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed4c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        
        self.embed_lin = torch.nn.Linear(self.size_embed*4, 64) 
        self.embed_log = torch.nn.Linear(self.size_embed*4, 64) 
        self.diff_lin = torch.nn.Linear(48*5*5, 64) 
        self.diff_log = torch.nn.Linear(48*5*5, 64) 
        
        
        self.conv1a = torch.nn.Conv3d(1, ndf, kernel_size=2, stride=(2,1,1), padding=0, bias=False)
        self.ln1a = torch.nn.LayerNorm([24,24,24])
        self.conv1b = torch.nn.Conv3d(ndf, ndf, kernel_size=2, stride=(2,2,2), padding=0, bias=False)
        self.ln1b = torch.nn.LayerNorm([12,12,12])
        self.conv1c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=(2,2,2), padding=(1,1,1), bias=False)
        
        self.conv2a = torch.nn.Conv3d(1, ndf, kernel_size=2, stride=(2,1,1), padding=0, bias=False)
        self.ln2a = torch.nn.LayerNorm([24,24,24])
        self.conv2b = torch.nn.Conv3d(ndf, ndf, kernel_size=2, stride=(2,2,2), padding=0, bias=False)
        self.ln2b = torch.nn.LayerNorm([12,12,12])
        self.conv2c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=(2,2,2), padding=(1,1,1), bias=False)

        self.conv_lin = torch.nn.Linear(6*6*6*ndf, 64) 
        self.conv_log = torch.nn.Linear(6*6*6*ndf, 64) 
        
        self.econd_lin = torch.nn.Linear(1, 64) 

        self.fc1 = torch.nn.Linear(64*7, 256)
        self.fc2 = torch.nn.Linear(256,  256)
        self.fc3 = torch.nn.Linear(256, 1)


    def forward(self, img, E_true):
        batch_size = img.size(0)
        
        #img_log_a = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)
        #img_log_b = img[:,0:1,:,:,:] #data.view(-1,1,30,30)

        #print('img_log_a', torch.sum(img_log_a))
        #print('img_log_b', torch.sum(img_log_b))

        #img_a = torch.exp(img_log_a-5.0)-np.exp(-5.0)
        #img_b = torch.exp(img_log_b-5.0)-np.exp(-5.0)
        
        img_a = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)
        img_b = img[:,0:1,:,:,:] #data.view(-1,1,30,30)
        
        img_log_a = torch.log(img_a+np.exp(-5.0))+5.0
        img_log_b = torch.log(img_b+np.exp(-5.0))+5.0
        
        
        img_a_hit_std = torch.std(img_a.view(batch_size, -1), 1)
        img_a_hit_std_a = img_a_hit_std.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_a_hit_std_b = img_a_hit_std.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_a_hit_std = img_a_hit_std_a - img_a_hit_std_b
        
        img_a_hit_sum = torch.sum(img_a.view(batch_size, -1), 1)
        img_a_hit_sum_a = img_a_hit_sum.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_a_hit_sum_b = img_a_hit_sum.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_a_hit_sum = img_a_hit_sum_a - img_a_hit_sum_b

        
        img_a_hit_std = torch.transpose(img_a_hit_std, 1, 2)
        img_a_hit_sum = torch.transpose(img_a_hit_sum, 1, 2)

        
        img_a_hit_std = F.leaky_relu(self.Embed1a(img_a_hit_std))
        img_a_hit_std = F.leaky_relu(self.Embed1b(img_a_hit_std))
        img_a_hit_std = F.leaky_relu(self.Embed1c(img_a_hit_std))
        
        img_a_hit_sum = F.leaky_relu(self.Embed2a(img_a_hit_sum))
        img_a_hit_sum = F.leaky_relu(self.Embed2b(img_a_hit_sum))
        img_a_hit_sum = F.leaky_relu(self.Embed2c(img_a_hit_sum))

        img_a_hit_std_std  =  torch.std(img_a_hit_std, 2)
        img_a_hit_std_mean = torch.mean(img_a_hit_std, 2)

        img_a_hit_sum_std  =  torch.std(img_a_hit_sum, 2)
        img_a_hit_sum_mean = torch.mean(img_a_hit_sum, 2)
        
        xa1 = torch.cat((img_a_hit_std_std, img_a_hit_std_mean, img_a_hit_sum_std, img_a_hit_sum_mean), 1)
        xa1 = F.leaky_relu(self.embed_lin(xa1), 0.2)
        
        
        img_log_a_hit_std = torch.std(img_log_a.view(batch_size, -1), 1)
        img_log_a_hit_std_a = img_log_a_hit_std.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_log_a_hit_std_b = img_log_a_hit_std.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_log_a_hit_std = img_log_a_hit_std_a - img_log_a_hit_std_b
        
        img_log_a_hit_sum = torch.sum(img_log_a.view(batch_size, -1), 1)
        img_log_a_hit_sum_a = img_log_a_hit_sum.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_log_a_hit_sum_b = img_log_a_hit_sum.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_log_a_hit_sum = img_log_a_hit_sum_a - img_log_a_hit_sum_b
        
        img_log_a_hit_std = torch.transpose(img_log_a_hit_std, 1, 2)
        img_log_a_hit_sum = torch.transpose(img_log_a_hit_sum, 1, 2)
        
        img_log_a_hit_std = F.leaky_relu(self.Embed1a(img_log_a_hit_std))
        img_log_a_hit_std = F.leaky_relu(self.Embed1b(img_log_a_hit_std))
        img_log_a_hit_std = F.leaky_relu(self.Embed1c(img_log_a_hit_std))
        
        img_log_a_hit_sum = F.leaky_relu(self.Embed2a(img_log_a_hit_sum))
        img_log_a_hit_sum = F.leaky_relu(self.Embed2b(img_log_a_hit_sum))
        img_log_a_hit_sum = F.leaky_relu(self.Embed2c(img_log_a_hit_sum))

        img_log_a_hit_std_std  =  torch.std(img_log_a_hit_std, 2)
        img_log_a_hit_std_mean = torch.mean(img_log_a_hit_std, 2)

        img_log_a_hit_sum_std  =  torch.std(img_log_a_hit_sum, 2)
        img_log_a_hit_sum_mean = torch.mean(img_log_a_hit_sum, 2)        
        
        xa2 = torch.cat((img_log_a_hit_std_std, img_log_a_hit_std_mean, img_log_a_hit_sum_std, img_log_a_hit_sum_mean), 1)
        xa2 = F.leaky_relu(self.embed_lin(xa2), 0.2)
        
               
        xa3 = F.leaky_relu(self.diff_lin((img_a[:,:,:,10:15,10:15]-img_b[:,:,:,10:15,10:15]).view(batch_size, -1)), 0.2)
        
        
        xa4 = F.leaky_relu(self.diff_log((img_log_a[:,:,:,10:15,10:15]-img_log_b[:,:,:,10:15,10:15]).view(batch_size, -1)), 0.2)
 

        xa5 = F.leaky_relu(self.ln1a(self.conv1a(img_a*0.001)), 0.2)
        xa5 = F.leaky_relu(self.ln1b(self.conv1b(xa5)), 0.2)
        xa5 = F.leaky_relu(self.conv1c(xa5), 0.2)
        #print(xa5.size())
        xa5 = xa5.view(-1, self.ndf*6*6*6)
        xa5 = F.leaky_relu(self.conv_lin(xa5), 0.2)
        

        xa6 = F.leaky_relu(self.ln2a(self.conv2a(img_log_a)), 0.2)
        xa6 = F.leaky_relu(self.ln2b(self.conv2b(xa6)), 0.2)
        xa6 = F.leaky_relu(self.conv2c(xa6), 0.2)
        #print(xa6.size())
        xa6 = xa6.view(-1, self.ndf*6*6*6)
        xa6 = F.leaky_relu(self.conv_log(xa6), 0.2)        
        
        xa7 = F.leaky_relu(self.econd_lin(E_true), 0.2)      
        
        xa = torch.cat((xa1, xa2, xa3, xa4, xa5, xa6, xa7), 1)
        
        #print('xa1', torch.sum(xa1))
        #print('xa2', torch.sum(xa2))
        #print('xa3', torch.sum(xa3))
        #print('xa4', torch.sum(xa4))
        #print('xa5', torch.sum(xa5))
        #print('xa6', torch.sum(xa6))
        #print('xa7', torch.sum(xa7))
        
        xa = F.leaky_relu(self.fc1(xa), 0.2)
        xa = F.leaky_relu(self.fc2(xa), 0.2)
        xa = self.fc3(xa)

        #print('xa', torch.sum(xa))
        
        return xa ### flattens
    
    
    

    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
   
    
class Discriminator_F_BatchStat_Core25(nn.Module):
    def __init__(self, isize=48, nc=2, ndf=64):
        super(Discriminator_F_BatchStat_Core25, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc
        self.size_embed = 16
        self.conv1_bias = False

        self.Embed1a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed1b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed1c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed2a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed2b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed2c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed3a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed3b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed3c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed4a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed4b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed4c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        
        self.embed_lin = torch.nn.Linear(self.size_embed*4, 64) 
        self.embed_log = torch.nn.Linear(self.size_embed*4, 64) 
        self.diff_lin = torch.nn.Linear(48*25*25, 64) 
        self.diff_log = torch.nn.Linear(48*25*25, 64) 
        
        
        self.conv1a = torch.nn.Conv3d(1, ndf, kernel_size=2, stride=(2,1,1), padding=0, bias=False)
        self.ln1a = torch.nn.LayerNorm([24,24,24])
        self.conv1b = torch.nn.Conv3d(ndf, ndf, kernel_size=2, stride=(2,2,2), padding=0, bias=False)
        self.ln1b = torch.nn.LayerNorm([12,12,12])
        self.conv1c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=(2,2,2), padding=(1,1,1), bias=False)
        
        self.conv2a = torch.nn.Conv3d(1, ndf, kernel_size=2, stride=(2,1,1), padding=0, bias=False)
        self.ln2a = torch.nn.LayerNorm([24,24,24])
        self.conv2b = torch.nn.Conv3d(ndf, ndf, kernel_size=2, stride=(2,2,2), padding=0, bias=False)
        self.ln2b = torch.nn.LayerNorm([12,12,12])
        self.conv2c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=(2,2,2), padding=(1,1,1), bias=False)

        self.conv_lin = torch.nn.Linear(6*6*6*ndf, 64) 
        self.conv_log = torch.nn.Linear(6*6*6*ndf, 64) 
        
        self.econd_lin = torch.nn.Linear(1, 64) 

        self.fc1 = torch.nn.Linear(64*7, 256)
        self.fc2 = torch.nn.Linear(256,  256)
        self.fc3 = torch.nn.Linear(256, 1)


    def forward(self, img, E_true):
        batch_size = img.size(0)
        
        #img_log_a = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)
        #img_log_b = img[:,0:1,:,:,:] #data.view(-1,1,30,30)

        #print('img_log_a', torch.sum(img_log_a))
        #print('img_log_b', torch.sum(img_log_b))

        #img_a = torch.exp(img_log_a-5.0)-np.exp(-5.0)
        #img_b = torch.exp(img_log_b-5.0)-np.exp(-5.0)
        
        img_a = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)
        img_b = img[:,0:1,:,:,:] #data.view(-1,1,30,30)
        
        img_log_a = torch.log(img_a+np.exp(-5.0))+5.0
        img_log_b = torch.log(img_b+np.exp(-5.0))+5.0
        
        
        img_a_hit_std = torch.std(img_a.view(batch_size, -1), 1)
        img_a_hit_std_a = img_a_hit_std.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_a_hit_std_b = img_a_hit_std.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_a_hit_std = img_a_hit_std_a - img_a_hit_std_b
        
        img_a_hit_sum = torch.sum(img_a.view(batch_size, -1), 1)
        img_a_hit_sum_a = img_a_hit_sum.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_a_hit_sum_b = img_a_hit_sum.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_a_hit_sum = img_a_hit_sum_a - img_a_hit_sum_b

        
        img_a_hit_std = torch.transpose(img_a_hit_std, 1, 2)
        img_a_hit_sum = torch.transpose(img_a_hit_sum, 1, 2)

        
        img_a_hit_std = F.leaky_relu(self.Embed1a(img_a_hit_std))
        img_a_hit_std = F.leaky_relu(self.Embed1b(img_a_hit_std))
        img_a_hit_std = F.leaky_relu(self.Embed1c(img_a_hit_std))
        
        img_a_hit_sum = F.leaky_relu(self.Embed2a(img_a_hit_sum))
        img_a_hit_sum = F.leaky_relu(self.Embed2b(img_a_hit_sum))
        img_a_hit_sum = F.leaky_relu(self.Embed2c(img_a_hit_sum))

        img_a_hit_std_std  =  torch.std(img_a_hit_std, 2)
        img_a_hit_std_mean = torch.mean(img_a_hit_std, 2)

        img_a_hit_sum_std  =  torch.std(img_a_hit_sum, 2)
        img_a_hit_sum_mean = torch.mean(img_a_hit_sum, 2)
        
        xa1 = torch.cat((img_a_hit_std_std, img_a_hit_std_mean, img_a_hit_sum_std, img_a_hit_sum_mean), 1)
        xa1 = F.leaky_relu(self.embed_lin(xa1), 0.2)
        
        
        img_log_a_hit_std = torch.std(img_log_a.view(batch_size, -1), 1)
        img_log_a_hit_std_a = img_log_a_hit_std.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_log_a_hit_std_b = img_log_a_hit_std.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_log_a_hit_std = img_log_a_hit_std_a - img_log_a_hit_std_b
        
        img_log_a_hit_sum = torch.sum(img_log_a.view(batch_size, -1), 1)
        img_log_a_hit_sum_a = img_log_a_hit_sum.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_log_a_hit_sum_b = img_log_a_hit_sum.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_log_a_hit_sum = img_log_a_hit_sum_a - img_log_a_hit_sum_b
        
        img_log_a_hit_std = torch.transpose(img_log_a_hit_std, 1, 2)
        img_log_a_hit_sum = torch.transpose(img_log_a_hit_sum, 1, 2)
        
        img_log_a_hit_std = F.leaky_relu(self.Embed1a(img_log_a_hit_std))
        img_log_a_hit_std = F.leaky_relu(self.Embed1b(img_log_a_hit_std))
        img_log_a_hit_std = F.leaky_relu(self.Embed1c(img_log_a_hit_std))
        
        img_log_a_hit_sum = F.leaky_relu(self.Embed2a(img_log_a_hit_sum))
        img_log_a_hit_sum = F.leaky_relu(self.Embed2b(img_log_a_hit_sum))
        img_log_a_hit_sum = F.leaky_relu(self.Embed2c(img_log_a_hit_sum))

        img_log_a_hit_std_std  =  torch.std(img_log_a_hit_std, 2)
        img_log_a_hit_std_mean = torch.mean(img_log_a_hit_std, 2)

        img_log_a_hit_sum_std  =  torch.std(img_log_a_hit_sum, 2)
        img_log_a_hit_sum_mean = torch.mean(img_log_a_hit_sum, 2)        
        
        xa2 = torch.cat((img_log_a_hit_std_std, img_log_a_hit_std_mean, img_log_a_hit_sum_std, img_log_a_hit_sum_mean), 1)
        xa2 = F.leaky_relu(self.embed_lin(xa2), 0.2)
        
               
        xa3 = F.leaky_relu(self.diff_lin((img_a-img_b).view(batch_size, -1)), 0.2)
        
        
        xa4 = F.leaky_relu(self.diff_log((img_log_a-img_log_b).view(batch_size, -1)), 0.2)
 

        xa5 = F.leaky_relu(self.ln1a(self.conv1a(img_a*0.001)), 0.2)
        xa5 = F.leaky_relu(self.ln1b(self.conv1b(xa5)), 0.2)
        xa5 = F.leaky_relu(self.conv1c(xa5), 0.2)
        #print(xa5.size())
        xa5 = xa5.view(-1, self.ndf*6*6*6)
        xa5 = F.leaky_relu(self.conv_lin(xa5), 0.2)
        

        xa6 = F.leaky_relu(self.ln2a(self.conv2a(img_log_a)), 0.2)
        xa6 = F.leaky_relu(self.ln2b(self.conv2b(xa6)), 0.2)
        xa6 = F.leaky_relu(self.conv2c(xa6), 0.2)
        #print(xa6.size())
        xa6 = xa6.view(-1, self.ndf*6*6*6)
        xa6 = F.leaky_relu(self.conv_log(xa6), 0.2)        
        
        xa7 = F.leaky_relu(self.econd_lin(E_true), 0.2)      
        
        xa = torch.cat((xa1, xa2, xa3, xa4, xa5, xa6, xa7), 1)
        
        #print('xa1', torch.sum(xa1))
        #print('xa2', torch.sum(xa2))
        #print('xa3', torch.sum(xa3))
        #print('xa4', torch.sum(xa4))
        #print('xa5', torch.sum(xa5))
        #print('xa6', torch.sum(xa6))
        #print('xa7', torch.sum(xa7))
        
        xa = F.leaky_relu(self.fc1(xa), 0.2)
        xa = F.leaky_relu(self.fc2(xa), 0.2)
        xa = self.fc3(xa)

        #print('xa', torch.sum(xa))
        
        return xa ### flattens
    
    
    

    
    
    
    
    
    
    
    
class BiBAE_F_3D_BatchStat_Core25(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, device, nc=1, ngf=8, z_rand=500, z_enc=12):
        super(BiBAE_F_3D_BatchStat_Core25, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        self.z_rand = z_rand
        self.z_enc = z_enc
        self.z_full = z_enc + z_rand
        self.args = args
        self.device = device

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(2,2,2), stride=(2,1,1),
                               padding=(0,0,0), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.LayerNorm([24,24,24])
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(2,2,2), stride=(2,2,2),
                               padding=(0,0,0), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.LayerNorm([12,12,12])
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.LayerNorm([6,6,6])
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*4, kernel_size=(4,4,4), stride=(1,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.LayerNorm([5,5,5])

     
        self.fc1 = nn.Linear(5*5*5*ngf*4+1, ngf*100, bias=True)
        self.fc2 = nn.Linear(ngf*100, int(self.z_full), bias=True)
        
        self.fc31 = nn.Linear(int(self.z_full), self.z_enc, bias=True)
        self.fc32 = nn.Linear(int(self.z_full), self.z_enc, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z_full+1, int(self.z_full), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z_full), ngf*100, bias=True)
        self.cond3 = torch.nn.Linear(ngf*100, 7*7*12*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.LayerNorm([24,14,14])
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.LayerNorm([48,28,28])

        self.conv0a = torch.nn.Conv3d(ngf, ngf, kernel_size=(1,2,2), stride=(1,1,1), padding=(0,0,0), bias=False)
        self.bnco0a = torch.nn.LayerNorm([48,27,27])
        self.conv0b = torch.nn.Conv3d(ngf, ngf, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,0,0), bias=False)
        self.bnco0b = torch.nn.LayerNorm([48,25,25])
        self.conv0 = torch.nn.Conv3d(ngf, ngf, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco0 = torch.nn.LayerNorm([48,25,25])
        self.conv1 = torch.nn.Conv3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco1 = torch.nn.LayerNorm([48,25,25])
        self.conv2 = torch.nn.Conv3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco2 = torch.nn.LayerNorm([48,25,25])
        self.conv3 = torch.nn.Conv3d(ngf*4, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.bnco3 = torch.nn.LayerNorm([48,25,25])
        self.conv4 = torch.nn.Conv3d(ngf*2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)


    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,48,25,25))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        return torch.cat((self.fc31(x),torch.zeros(x.size(0), self.z_rand, device = x.device)), 1), torch.cat((self.fc32(x),torch.zeros(x.size(0), self.z_rand, device = x.device)), 1)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)

        x = x.view(-1,self.ngf,12,7,7)        

        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 24, 14, 14])), 0.2, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 48, 28, 28])), 0.2, inplace=True) #


        x = F.leaky_relu(self.bnco0a(self.conv0a(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco0b(self.conv0b(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco3(self.conv3(x)), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        return x 
        
        
    def forward(self, x, E_true, z=None, mode='full'):
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z 
        elif mode == 'decode':
            return self.decode(torch.cat((z,E_true), 1)) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), mu, logvar, z
        elif mode == 'half':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1))
            
    
    

    
    
    
    
    
    

    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
   
    
class Discriminator_F_BatchStat_Core25_larger(nn.Module):
    def __init__(self, isize=48, nc=2, ndf=64):
        super(Discriminator_F_BatchStat_Core25_larger, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc
        self.size_embed = 16
        self.conv1_bias = False

        self.Embed1a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed1b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed1c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed2a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed2b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed2c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed3a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed3b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed3c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed4a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed4b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed4c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        
        self.embed_lin = torch.nn.Linear(self.size_embed*4, 64) 
        self.embed_log = torch.nn.Linear(self.size_embed*4, 64) 
        self.diff_lin = torch.nn.Linear(48*25*25, 64) 
        self.diff_log = torch.nn.Linear(48*25*25, 64) 
        
        
        self.conv1 = torch.nn.Conv3d(1, ndf, kernel_size=3, stride=(1,1,1), padding=1, bias=False)
        self.ln1 = torch.nn.LayerNorm([48,25,25])
        self.conv1a = torch.nn.Conv3d(ndf, ndf, kernel_size=2, stride=(2,1,1), padding=0, bias=False)
        self.ln1a = torch.nn.LayerNorm([24,24,24])
        self.conv1b = torch.nn.Conv3d(ndf, ndf, kernel_size=2, stride=(2,2,2), padding=0, bias=False)
        self.ln1b = torch.nn.LayerNorm([12,12,12])
        self.conv1c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=(2,2,2), padding=(1,1,1), bias=False)

        self.conv2 = torch.nn.Conv3d(1, ndf, kernel_size=3, stride=(1,1,1), padding=1, bias=False)
        self.ln2 = torch.nn.LayerNorm([48,25,25])
        self.conv2a = torch.nn.Conv3d(ndf, ndf, kernel_size=2, stride=(2,1,1), padding=0, bias=False)
        self.ln2a = torch.nn.LayerNorm([24,24,24])
        self.conv2b = torch.nn.Conv3d(ndf, ndf, kernel_size=2, stride=(2,2,2), padding=0, bias=False)
        self.ln2b = torch.nn.LayerNorm([12,12,12])
        self.conv2c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=(2,2,2), padding=(1,1,1), bias=False)

        self.conv_lin = torch.nn.Linear(6*6*6*ndf, 64) 
        self.conv_log = torch.nn.Linear(6*6*6*ndf, 64) 
        
        self.econd_lin = torch.nn.Linear(1, 64) 

        self.fc1 = torch.nn.Linear(64*7, 256)
        self.fc2 = torch.nn.Linear(256,  256)
        self.fc3 = torch.nn.Linear(256, 1)


    def forward(self, img, E_true):
        batch_size = img.size(0)
        
        #img_log_a = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)
        #img_log_b = img[:,0:1,:,:,:] #data.view(-1,1,30,30)

        #print('img_log_a', torch.sum(img_log_a))
        #print('img_log_b', torch.sum(img_log_b))

        #img_a = torch.exp(img_log_a-5.0)-np.exp(-5.0)
        #img_b = torch.exp(img_log_b-5.0)-np.exp(-5.0)
        
        img_a = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)
        img_b = img[:,0:1,:,:,:] #data.view(-1,1,30,30)
        
        img_log_a = torch.log(img_a+np.exp(-5.0))+5.0
        img_log_b = torch.log(img_b+np.exp(-5.0))+5.0
        
        
        img_a_hit_std = torch.std(img_a.view(batch_size, -1), 1)
        img_a_hit_std_a = img_a_hit_std.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_a_hit_std_b = img_a_hit_std.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_a_hit_std = img_a_hit_std_a - img_a_hit_std_b
        
        img_a_hit_sum = torch.sum(img_a.view(batch_size, -1), 1)
        img_a_hit_sum_a = img_a_hit_sum.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_a_hit_sum_b = img_a_hit_sum.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_a_hit_sum = img_a_hit_sum_a - img_a_hit_sum_b

        
        img_a_hit_std = torch.transpose(img_a_hit_std, 1, 2)
        img_a_hit_sum = torch.transpose(img_a_hit_sum, 1, 2)

        
        img_a_hit_std = F.leaky_relu(self.Embed1a(img_a_hit_std))
        img_a_hit_std = F.leaky_relu(self.Embed1b(img_a_hit_std))
        img_a_hit_std = F.leaky_relu(self.Embed1c(img_a_hit_std))
        
        img_a_hit_sum = F.leaky_relu(self.Embed2a(img_a_hit_sum))
        img_a_hit_sum = F.leaky_relu(self.Embed2b(img_a_hit_sum))
        img_a_hit_sum = F.leaky_relu(self.Embed2c(img_a_hit_sum))

        img_a_hit_std_std  =  torch.std(img_a_hit_std, 2)
        img_a_hit_std_mean = torch.mean(img_a_hit_std, 2)

        img_a_hit_sum_std  =  torch.std(img_a_hit_sum, 2)
        img_a_hit_sum_mean = torch.mean(img_a_hit_sum, 2)
        
        xa1 = torch.cat((img_a_hit_std_std, img_a_hit_std_mean, img_a_hit_sum_std, img_a_hit_sum_mean), 1)
        xa1 = F.leaky_relu(self.embed_lin(xa1), 0.2)
        
        
        img_log_a_hit_std = torch.std(img_log_a.view(batch_size, -1), 1)
        img_log_a_hit_std_a = img_log_a_hit_std.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_log_a_hit_std_b = img_log_a_hit_std.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_log_a_hit_std = img_log_a_hit_std_a - img_log_a_hit_std_b
        
        img_log_a_hit_sum = torch.sum(img_log_a.view(batch_size, -1), 1)
        img_log_a_hit_sum_a = img_log_a_hit_sum.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_log_a_hit_sum_b = img_log_a_hit_sum.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_log_a_hit_sum = img_log_a_hit_sum_a - img_log_a_hit_sum_b
        
        img_log_a_hit_std = torch.transpose(img_log_a_hit_std, 1, 2)
        img_log_a_hit_sum = torch.transpose(img_log_a_hit_sum, 1, 2)
        
        img_log_a_hit_std = F.leaky_relu(self.Embed1a(img_log_a_hit_std))
        img_log_a_hit_std = F.leaky_relu(self.Embed1b(img_log_a_hit_std))
        img_log_a_hit_std = F.leaky_relu(self.Embed1c(img_log_a_hit_std))
        
        img_log_a_hit_sum = F.leaky_relu(self.Embed2a(img_log_a_hit_sum))
        img_log_a_hit_sum = F.leaky_relu(self.Embed2b(img_log_a_hit_sum))
        img_log_a_hit_sum = F.leaky_relu(self.Embed2c(img_log_a_hit_sum))

        img_log_a_hit_std_std  =  torch.std(img_log_a_hit_std, 2)
        img_log_a_hit_std_mean = torch.mean(img_log_a_hit_std, 2)

        img_log_a_hit_sum_std  =  torch.std(img_log_a_hit_sum, 2)
        img_log_a_hit_sum_mean = torch.mean(img_log_a_hit_sum, 2)        
        
        xa2 = torch.cat((img_log_a_hit_std_std, img_log_a_hit_std_mean, img_log_a_hit_sum_std, img_log_a_hit_sum_mean), 1)
        xa2 = F.leaky_relu(self.embed_lin(xa2), 0.2)
        
               
        xa3 = F.leaky_relu(self.diff_lin((img_a-img_b).view(batch_size, -1)), 0.2)
        
        
        xa4 = F.leaky_relu(self.diff_log((img_log_a-img_log_b).view(batch_size, -1)), 0.2)
 

        xa5 = F.leaky_relu(self.ln1(self.conv1(img_a*0.001)), 0.2)
        xa5 = F.leaky_relu(self.ln1a(self.conv1a(xa5)), 0.2)
        xa5 = F.leaky_relu(self.ln1b(self.conv1b(xa5)), 0.2)
        xa5 = F.leaky_relu(self.conv1c(xa5), 0.2)
        #print(xa5.size())
        xa5 = xa5.view(-1, self.ndf*6*6*6)
        xa5 = F.leaky_relu(self.conv_lin(xa5), 0.2)
        

        xa6 = F.leaky_relu(self.ln2(self.conv2(img_log_a)), 0.2)
        xa6 = F.leaky_relu(self.ln2a(self.conv2a(xa6)), 0.2)
        xa6 = F.leaky_relu(self.ln2b(self.conv2b(xa6)), 0.2)
        xa6 = F.leaky_relu(self.conv2c(xa6), 0.2)
        #print(xa6.size())
        xa6 = xa6.view(-1, self.ndf*6*6*6)
        xa6 = F.leaky_relu(self.conv_log(xa6), 0.2)        
        
        xa7 = F.leaky_relu(self.econd_lin(E_true), 0.2)      
        
        xa = torch.cat((xa1, xa2, xa3, xa4, xa5, xa6, xa7), 1)
        
        #print('xa1', torch.sum(xa1))
        #print('xa2', torch.sum(xa2))
        #print('xa3', torch.sum(xa3))
        #print('xa4', torch.sum(xa4))
        #print('xa5', torch.sum(xa5))
        #print('xa6', torch.sum(xa6))
        #print('xa7', torch.sum(xa7))
        
        xa = F.leaky_relu(self.fc1(xa), 0.2)
        xa = F.leaky_relu(self.fc2(xa), 0.2)
        xa = self.fc3(xa)

        #print('xa', torch.sum(xa))
        
        return xa ### flattens
    
    
    

    
    
class PostProcess_Size1Conv_EcondV2_Core25(nn.Module):
    def __init__(self, isize=48, isize2=25, nc=2, ndf=64, bias=False, out_funct='relu'):
        super(PostProcess_Size1Conv_EcondV2_Core25, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.isize2 = isize2
        self.nc = nc
        self.bais = bias
        self.out_funct = out_funct
        
        self.fcec1 = torch.nn.Linear(2, int(ndf/2), bias=True)
        self.fcec2 = torch.nn.Linear(int(ndf/2), int(ndf/2), bias=True)
        self.fcec3 = torch.nn.Linear(int(ndf/2), int(ndf/2), bias=True)
        
        self.conv1 = torch.nn.Conv3d(1, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco1 = torch.nn.LayerNorm([self.isize, self.isize2, self.isize2])
        
        self.conv2 = torch.nn.Conv3d(ndf+int(ndf/2), ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco2 = torch.nn.LayerNorm([self.isize, self.isize2, self.isize2])

        self.conv3 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco3 = torch.nn.LayerNorm([self.isize, self.isize2, self.isize2])

        self.conv4 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco4 = torch.nn.LayerNorm([self.isize, self.isize2, self.isize2])

        self.conv5 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)

        self.conv6 = torch.nn.Conv3d(ndf, 1, kernel_size=1, stride=1, padding=0, bias=False)
 

    def forward(self, img, E_True=0):
        img = img.view(-1, 1, self.isize, self.isize2, self.isize2) 
                
        econd = torch.cat((torch.sum(img.view(-1, self.isize2*self.isize2*self.isize), 1).view(-1, 1), E_True), 1)

        econd = F.leaky_relu(self.fcec1(econd), 0.01)
        econd = F.leaky_relu(self.fcec2(econd), 0.01)
        econd = F.leaky_relu(self.fcec3(econd), 0.01)
        
        econd = econd.view(-1, int(self.ndf/2), 1, 1, 1)
        econd = econd.expand(-1, -1, self.isize, self.isize2, self.isize2)        
        
        img = F.leaky_relu(self.bnco1(self.conv1(img)), 0.01)
        img = torch.cat((img, econd), 1)
        img = F.leaky_relu(self.bnco2(self.conv2(img)), 0.01)
        img = F.leaky_relu(self.bnco3(self.conv3(img)), 0.01)
        img = F.leaky_relu(self.bnco4(self.conv4(img)), 0.01)
        img = F.leaky_relu(self.conv5(img), 0.01) 
        img = self.conv6(img)

        if self.out_funct == 'relu':
            img = F.relu(img)
        elif self.out_funct == 'leaky_relu':
            img = F.leaky_relu(img, 0.01) 
              
        return img.view(-1, 1, self.isize, self.isize2, self.isize2)


     
    