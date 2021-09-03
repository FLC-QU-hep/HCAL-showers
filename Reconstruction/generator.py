import numpy as np
import argparse
import torch
import array as arr
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import models.dcgan3Dcore25 as WGAN_Models
from models.dcgan3Dcore25 import *
from models.bibae import *


import streamlit as st
#import pkbar
import time
import os 
import dill as pickle
import random

import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.use('GTKAgg')


'''
# Synthetic Shower Generation 

Shower simulation is CPU intensive. Generative machine learning models based on deep neural networks are speeding up this task by several orders of magnitude

## Generative Models:
*  Generative Adversarial Network (GAN)
*  Wasserstein Generative Adversarial Network (WGAN) 
*  Bounded Information Bottleneck Variational Autoendoder (BiB-AE)

'''



from pyLCIO import EVENT, UTIL, IOIMPL, IMPL

def get_parser():
    parser = argparse.ArgumentParser(
        description='Generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--nbsize', action='store',
                        type=int, default=10,
                        help='Batch size for generation')

    parser.add_argument('--model', action='store',
                        type=str, default="wgan",
                        help='type of model (bib-ae , wgan or gan)')

    parser.add_argument('--output', action='store',
                        type=str, help='Name of the output file')



    return parser

def getTotE(data, xbins, ybins, layers=30):
    data = np.reshape(data,[-1, layers*xbins*ybins])
    etot_arr = np.sum(data, axis=(1))
    return etot_arr


def correctSpinalProfileFake(data, xbins, ybins, layers):
    data = np.reshape(data,[-1, layers, xbins*ybins])
    #correct layers 8-15 by a factor 
    data[:,8:18,:] = data[:,8:18,:] * 0.82
    data = np.reshape(data,[-1, layers, xbins, ybins])
    return data


def calibrate(data, xbins, ybins, layers, sfac=1.00):
    data = np.reshape(data,[-1, layers, xbins*ybins]) 
    data[:,:,:] = data[:,:,:] * sfac
    data = np.reshape(data,[-1, layers, xbins, ybins])
    return data

def make_plots(fake_data):
    
    status_text = st.empty()
    status_text.text("Making some plots...")

    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    etot_fake = getTotE(fake_data, xbins=30, ybins=30)

    n, bins, patches = axs.hist(etot_fake.flatten(), bins=25, range=[0, 1500],
               weights=np.ones_like(etot_fake)/(float(len(etot_fake))),
               label = "Simulated hits", color= 'blue',
               histtype='stepfilled')
    

    st.pyplot(fig)


def lat_opt_ngd(G,D,z, energy, batch_size, device, alpha=500, beta=0.1, norm=1000):
    
    z.requires_grad_(True)
    x_hat = G(z, energy)
    x_hat = x_hat.unsqueeze(1) 
    
    f_z = D(x_hat, energy)

    fz_dz = torch.autograd.grad(outputs=f_z,
                                inputs= z,
                                grad_outputs=torch.ones(f_z.size()).to(device),
                                retain_graph=True,
                                create_graph= True,
                                   )[0]
    
    delta_z = torch.ones_like(fz_dz)
    delta_z = (alpha * fz_dz) / (beta +  torch.norm(delta_z, p=2, dim=0) / norm)
    with torch.no_grad():
        z_prime = torch.clamp(z + delta_z, min=-1, max=1) 
        
    return z_prime

def shower_particles(nevents, model, bsize, emax, emin): 
    
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    if model == 'wgan':
        
        LATENT_DIM = 100
        ngf = 32
        ndf = 32
        model_WGAN = WGAN_Models.DCGAN_G(ngf,LATENT_DIM).to(device)
        model_WGAN = nn.DataParallel(model_WGAN)
        weightsGAN = 'weights/wgan.pth'
        checkpoint = torch.load(weightsGAN, map_location=torch.device(device))
        model_WGAN.load_state_dict(checkpoint['Generator'])
        
        
        showers, energy = wGAN_ENR(model_WGAN, nevents, emax, emin, bsize, LATENT_DIM, device)
        energy = energy.flatten()

    else:
        
        LATENT_DIM_SML = 24
        LATENT_DIM = 512
        args = {
                'E_cond' : True,
                'latent' : LATENT_DIM
        }

        checkpoint_BAE = torch.load('weights/bibae.pth', map_location=torch.device(device))
        model_BAE = BiBAE_F_3D_BatchStat_Core25(args, device=device, z_rand=LATENT_DIM-LATENT_DIM_SML, z_enc=LATENT_DIM_SML).to(device)   
        model_BAE = nn.DataParallel(model_BAE)
        model_BAE.load_state_dict(checkpoint_BAE['model_state_dict'])

        model_BAE27_P = PostProcess_Size1Conv_EcondV2_Core25(bias=True, out_funct='none').to(device)
        model_BAE27_P = nn.DataParallel(model_BAE27_P)
        model_BAE27_P.load_state_dict(checkpoint_BAE['model_P_state_dict'])

        file='weights/Latent_Data_New_KDE_ep17.pkl'
        with open(file, 'rb') as f:
            kde_BAE27 = pickle.load(f)
        
        showers, energy = bibAE(model_BAE, model_BAE27_P, nevents, emax/100.0, emin/100.0, LATENT_DIM, kde_BAE27,  
                                     device='cpu', thresh = 0.00 )
        energy = energy.flatten()


    
    return showers, energy



def wGAN_ENR(model, number, E_max, E_min, batchsize, latent_dim, device):

    thresh = 0.5*0.5
    fake_list=[]
    energy_list = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Generating {} pion showers. Current models is WGAN".format(number))

    model.eval()
    for i in np.arange(0, number, batchsize):
        with torch.no_grad():
           
            
            noise = torch.FloatTensor(batchsize, latent_dim, 1,1,1).uniform_(-1, 1)
            noise = noise.view(-1, latent_dim, 1,1,1)
            noise = noise.to(device)
            
            
            input_energy = torch.FloatTensor(batchsize ,1).to(device) 
            input_energy.resize_(batchsize,1,1,1,1).uniform_(E_min, E_max)


            fake = model(noise, input_energy)
            fake = fake.data.cpu().numpy()
            
            #fake[fake < thresh] = 0.0
            fake_list.append(fake)
            energy_list.append(input_energy.data.cpu().numpy())

            
            progress_bar.progress(int(i* 100 / number))            
            


    energy_full = np.vstack(energy_list)
    fake_full = np.vstack(fake_list)
    
    if E_max != E_min: 
     
        fake_full = fake_full.reshape(len(fake_full), 48, 25, 25)
        return fake_full, energy_full
    
    else:
        
        with open('scale_factors.txt') as f:
            arr = []
            for line in f:
                arr.append([float(x) for x in line.split()])
        
        arrnp = np.asarray(arr)
        idx = np.where(arrnp[:,0] == round(E_max,2))
        sf = arrnp[idx[0][0]][1]
        #print("applying SF: ", sf, "to Energy: ",E_max, E_min )
        status_text.text("applying SF: {} to Energy: {}, {}".format(sf, E_max, E_min))
        fake_full_calb = calibrate(fake_full, xbins=25, ybins=25, layers=48, sfac=sf)
        return fake_full_calb, energy_full
    


    

    

def wGAN(model, model_aD, number, E_max, E_min, batchsize, fixed_noise, input_energy, device):


    #pbar = pkbar.Pbar(name='Generating {} photon showers with energies [{},{}]'.format(number, E_max,E_min), target=number)
    thresh = 0.5*0.5
    fake_list=[]
    energy_list = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Generating {} pion showers. Current models is WGAN-LO".format(number))

    for i in np.arange(batchsize, number+1, batchsize):
        
       
        input_energy.uniform_(E_min,E_max)
        #z_prime = lat_opt_ngd(model, model_aD, fixed_noise, input_energy, batchsize, device)
        
        with torch.no_grad():
           
            fake = model(fixed_noise, input_energy)
            fake = fake.data.cpu().numpy()
            
            fake[fake < thresh] = 0.0
            fake_list.append(fake)
            energy_list.append(input_energy.data.cpu().numpy())

            
            progress_bar.progress(int(i* 100 / number))            
            

    energy_full = np.vstack(energy_list)
    fake_full = np.vstack(fake_list)
    fake_full = fake_full.reshape(len(fake_full), 48, 25, 25)


    if (E_max+E_min)/2.0 < 70:
        fake_full_corr = correctSpinalProfileFake(fake_full, xbins=25, ybins=25, layers=48)
        #fake_full_corr = fake_full
        print ("turning corrections on")   

    elif (round(E_max,2)+round(E_min,2))/2.0 == 70:
        fake_full_corr = correctSpinalProfileFakeHighE(fake_full, xbins=25, ybins=25, layers=48, sfac=0.94)
        print ("correcting 70GeV")
    
    elif (round(E_max,2)+round(E_min,2))/2.0 == 80:
        fake_full_corr = correctSpinalProfileFakeHighE(fake_full, xbins=25, ybins=25, layers=48, sfac=0.98)
        print ("correcting 80GeV")
    
    elif (round(E_max,2)+round(E_min,2))/2.0 == 90:
        fake_full_corr = correctSpinalProfileFakeHighE(fake_full, xbins=25, ybins=25, layers=48, sfac=1.03)
        print ("correcting 90GeV")
    else:
        fake_full_corr = fake_full

    
    #progress_bar.empty()
    

    return fake_full_corr, energy_full


def vGAN(model, number, E_max, E_min, batchsize, fixed_noise, input_energy, device):


    #pbar = pkbar.Pbar(name='Generating {} photon showers with energies [{},{}]'.format(number, E_max,E_min), target=number)
    
    fake_list=[]
    energy_list = []
    
    model.eval()

    for i in np.arange(0, number, batchsize):
        with torch.no_grad():
            fixed_noise.uniform_(-1,1)
            input_energy.uniform_(E_min,E_max)
            fake = model(fixed_noise, input_energy)
            fake = fake.data.cpu().numpy()
            fake_list.append(fake)
            energy_list.append(input_energy.data.cpu().numpy())
            #pbar.update(i- 1 + batchsize)

    energy_full = np.vstack(energy_list)
    fake_full = np.vstack(fake_list)
    fake_full = fake_full.reshape(len(fake_full), 30, 30, 30)

    return fake_full, energy_full

def bibAE(model, model_PostProcess, number, E_max, E_min, latent_dim, kde,  
                                     device='cpu', thresh=0.0):
    
    z_size = 48
    x_size = 25
    y_size = 25
    
    progress_bar = st.progress(0)
    status_text = st.empty()


    batchsize = 100
    fake_list = []
    fakePP_list = []
    latent_list=[]
    fake_uncut_list = []
    E_list = []
    
    
    ## for modifiy one latent dimension with certain mean and std
    ## load kde kernel as pkl file (a probability density function)

    latent_sml = 24     # shape of the kde kernel / the small latent space dimension

    model.eval()
    model_PostProcess.eval()
    for i in np.arange(0, number, batchsize):
        with torch.no_grad():
            
            latent = []
            E = []
            numberAccepted = 0
            while numberAccepted < batchsize:
                kde_sample = kde.resample(10000).T
                kde_latent = kde_sample[:, :latent_sml]
                kde_energy = kde_sample[:, latent_sml:]
                
                energy_mask = (kde_energy[:,0] < (E_max*100.0+0.1)) & (kde_energy[:,0] > (E_min*100.0-0.1))  
                
                kde_latent_masked = kde_latent[energy_mask]
                kde_energy_masked = kde_energy[energy_mask]
                
                
                numberAccepted += kde_latent_masked.shape[0]
                
                normal_sample = torch.randn(kde_latent_masked.shape[0], latent_dim - latent_sml)
                latent.append(torch.cat((torch.from_numpy(np.float32(kde_latent_masked)), normal_sample), dim=1))
                E.append(torch.from_numpy(np.float32(kde_energy_masked)))
            
            
            latent = torch.cat((latent)).to(device)[:batchsize]
            E = torch.cat((E), 0).to(device)[:batchsize]
            x = torch.zeros(batchsize, latent_dim, device=device)
            z = latent
            
            
            data = model(x=x, E_true=E, 
                            z = z,  mode='decode')                
                        
            latent = torch.cat((z, E), dim=1)

            latent = latent.cpu().numpy()

            dataPP = model_PostProcess.forward(data, E)
            
            
            data = data.view(-1, z_size, x_size, y_size).cpu().numpy() 
            dataPP = dataPP.view(-1, z_size, x_size, y_size).cpu().numpy() 
            #print(ratio)
            progress_bar.progress(int(i* 100 / number))
           
        data_uncut = np.array(dataPP)
        data_uncut[ data_uncut < 0.005] = 0.0  
        fake_uncut_list.append(data_uncut)

        data[ data < thresh] = 0.0  
        dataPP[ dataPP < thresh] = 0.0  
        

        fake_list.append(data)
        E_list.append(E)
        fakePP_list.append(dataPP)
        latent_list.append(latent)
        #print(i)

    data_full = np.vstack(fake_list)
    dataPP_full = np.vstack(fakePP_list)
    latent_full = np.vstack(latent_list)
    E_full =  np.vstack(E_list)
    data_uncut_full = np.vstack(fake_uncut_list)

    #print(data_full.shape)
    return dataPP_full, E_full


def write_to_lcio(showers, energy, model_name, outfile, N):
    

    status_text = st.empty()
    
    status_text.text("Writing to LCIO files...")
    progress_bar = st.progress(0)

    ## get the dictionary
    f = open('cell_maps/cell-map_HCAL.pickle', 'rb')
    cmap = pickle.load(f)  
    
    #pbar_cache = pkbar.Pbar(name='Writing to lcio files', target=N)

    wrt = IOIMPL.LCFactory.getInstance().createLCWriter( )

    wrt.open( outfile , EVENT.LCIO.WRITE_NEW ) 

    random.seed()



    #========== MC particle properties ===================
    genstat  = 1
    charge = 1
    mass = 1.40e-01 
    #decayLen = 1.e32 
    pdg = 211


    # write a RunHeader
    run = IMPL.LCRunHeaderImpl() 
    run.setRunNumber( 0 ) 
    run.parameters().setValue("Generator", model_name)
    run.parameters().setValue("PDG", pdg )
    wrt.writeRunHeader( run ) 

    for j in range( 0, N ):

        ### MC particle Collections
        colmc = IMPL.LCCollectionVec( EVENT.LCIO.MCPARTICLE ) 

        ## we are shooting 90 deg. HCAL 
        px = 0.00 
        py = energy[j] 
        pz = 0.00 

        vx = 30.00
        vy = 1000.00
        vz = 1000.00

        epx = 50.00
        epy = 3000.00
        epz = 1000.00

        momentum = arr.array('f',[ px, py, pz ] )  
        vertex = arr.array('d',[vx,vy,vz])
        endpoint = arr.array('d', [epx,epy,epz])


        mcp = IMPL.MCParticleImpl() 
        mcp.setGeneratorStatus( genstat ) 
        mcp.setMass( mass )
        mcp.setPDG( pdg ) 
        mcp.setMomentum( momentum )
        mcp.setCharge( charge )
        mcp.setVertex(vertex)
        mcp.setEndpoint(endpoint)

        colmc.addElement( mcp )
        
        evt = IMPL.LCEventImpl() 
        evt.setEventNumber( j ) 
        evt.addCollection( colmc , "MCParticle" )


        ### Calorimeter Collections
        col = IMPL.LCCollectionVec( EVENT.LCIO.SIMCALORIMETERHIT ) 
        flag =  IMPL.LCFlagImpl(0) 
        flag.setBit( EVENT.LCIO.CHBIT_LONG )
        flag.setBit( EVENT.LCIO.CHBIT_ID1 )

        col.setFlag( flag.getFlag() )

        col.parameters().setValue(EVENT.LCIO.CellIDEncoding, 'system:0:5,module:5:3,stave:8:4,tower:12:5,layer:17:6,slice:23:4,x:32:-16,y:48:-16')
        evt.addCollection( col , "HcalBarrelRegCollection" )

        

        for layer in range(48):              ## loop over layers
            nx, nz = np.nonzero(showers[j][layer])   ## get non-zero energy cells  
            for k in range(0,len(nx)):
                try:
                    cell_energy = showers[j][layer][nx[k]][nz[k]] / 1000.0
                    tmp = cmap[(layer, nx[k], nz[k])]

                    sch = IMPL.SimCalorimeterHitImpl()

                    position = arr.array('f', [tmp[0],tmp[1],tmp[2]])
    
                    sch.setPosition(position)
                    sch.setEnergy(cell_energy)
                    sch.setCellID0(int(tmp[3]))
                    sch.setCellID1(int(tmp[4]))
                    col.addElement( sch )

                except KeyError:
                    # Key is not present
                    pass
                    
                                
        progress_bar.progress(int(j*100/N))
        
        wrt.writeEvent( evt ) 

    progress_bar.empty()
    st.info('LCIO file was created: {}'.format(os.getcwd() + '/'+outfile))
    status_text_rec = st.empty()
    status_text_rec.text("We are ready to run reconstruction via iLCsoft")
    wrt.close() 


if __name__ == "__main__":

    parser = get_parser()
    parse_args = parser.parse_args() 
    
    bsize = parse_args.nbsize

    model_name = parse_args.model
    

    

    wgan_check = st.checkbox('Generate pions with WGAN model')
    bibae_check = st.checkbox('Generate pions with BiB-AE model')
    
    nevts = st.slider( 'Number of pion showers', 100, 5000, 500, step=bsize)
    evalues = st.slider( 'Select a range of pion energies', 0, 100, (25, 75))


    emin=evalues[0]
    emax=evalues[1]

    output_lcio = parse_args.output

    if wgan_check:
        showers, energy = shower_particles(nevts, model_name, bsize, emax, emin)
        write_to_lcio(showers, energy, model_name, output_lcio, nevts)
    elif bibae_check:
        #showers, energy = shower_particles(nevts, model_name, bsize, emax, emin)
        sh = np.load('./singleEnergy/Showers_G4_90GeV_punch.npy')
        enr = np.load('./singleEnergy/Showers_Energy_ep141_90GeV.npy')
        showers = sh[:1750]
        energy = enr[:1750]
        write_to_lcio(showers, energy, model_name, output_lcio, 1750)    
    
    
    reco = st.checkbox('Run Reconstruction in iLCSoft')
    if reco and not os.path.exists(os.getcwd() + '/rec.lock'):
        os.mknod(os.getcwd() + '/rec.lock')

