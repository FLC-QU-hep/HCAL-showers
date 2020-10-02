from comet_ml import Experiment
import torch
import os, sys
sys.path.append(os.getcwd())
import functools
import argparse
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
#from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import pdb
from torch import nn
from torch import autograd
from torch import optim
from torch.autograd import grad
import torch.nn.init as init




## get/import models
from models.dcgan3D import *
from models.constrainer3D import *
from models.dataUtils import PionsDataset
#from models.postp import *




def calc_gradient_penalty(netD, real_data, fake_data, real_label, BATCH_SIZE, device, DIM):
    
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous()
    alpha = alpha.view(BATCH_SIZE, 1, 48, DIM, DIM)
    alpha = alpha.to(device)


    fake_data = fake_data.view(BATCH_SIZE, 1, 48, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)   

    disc_interpolates = netD(interpolates.float(), real_label.float())
    #disc_interpolates = netD(interpolates.float())

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def mmd_hit_loss_cast_mean(recon_x, x, alpha=0.01): 
    # alpha = 1/2*sigma**2
    
    B = x.size(0)
    
    x_batch = x.view(B, -1)
    y_batch = recon_x.view(B, -1)

    x = x_batch.view(B,1,-1)
    y = y_batch.view(B,1,-1)

    #print (x.shape)
    xx = torch.matmul(torch.transpose(x,1,2),x) 
    yy = torch.matmul(torch.transpose(y,1,2),y)
    xy = torch.matmul(torch.transpose(y,1,2),x)
    
    rx = (torch.diagonal(xx, dim1=1, dim2=2).unsqueeze(1).expand_as(xx))
    ry = (torch.diagonal(yy, dim1=1, dim2=2).unsqueeze(1).expand_as(yy))
    
    K = torch.exp(- alpha * (torch.transpose(rx,1,2) + rx - 2*xx))
    L = torch.exp(- alpha * (torch.transpose(ry,1,2) + ry - 2*yy))
    P = torch.exp(- alpha * (torch.transpose(ry,1,2) + rx - 2*xy))

    out = (torch.mean(K, (1,2))+torch.mean(L, (1,2)) - 2*torch.mean(P, (1,2)))
    
    return out



def mmd_hit_sortKernel(recon_x_sorted, x_sorted, kernel_size, stride, cutoff, alpha = 200):
    
    B = x_sorted.size(0)
    pixels = x_sorted.size(1)
    out = 0
    norm_out = 0
    
    for j in np.arange(0, min(cutoff, pixels), step = stride):
        distx = x_sorted[:, j:j+kernel_size]
        disty = recon_x_sorted[:, j:j+kernel_size]

        if j == 0:
            out = mmd_hit_loss_cast_mean(disty, distx, alpha=alpha)
        else:
            out += mmd_hit_loss_cast_mean(disty, distx, alpha=alpha)
        
        norm_out += 1
    return (torch.mean(out)/norm_out)





def train(defparams, hyper):
    
    params = {}
    for param in defparams.keys():
        params[param] = defparams[param]

    hyperp = {}
    for hp in hyper.keys():
        hyperp[hp] = hyper[hp]

    experiment = Experiment(api_key="keGmeIz4GfKlQZlOP6cit4QOi",
                        project_name="hadron-shower", workspace="engineren")
    experiment.add_tag(params['exp'])

    experiment.log_parameters(hyperp)


    device = torch.device("cuda")       
    torch.manual_seed(params["seed"])

       
    aD = DCGAN_D(hyperp["ndf"]).to(device)
    aG = DCGAN_G(hyperp["ngf"], hyperp["z"]).to(device)
    aE = energyRegressor().to(device)
    #aP = PostProcess_Size1Conv_EcondV2(30, 3, 128, bias=True, out_funct='none').to(device)

    experiment.set_model_graph(str(aG))
    experiment.set_model_graph(str(aD))

    ## no need for post processing now
    #if params["restore_pp"]:
    #    aP.load_state_dict(torch.load(params["restore_path_PP"] + params["post_saved"], map_location=torch.device(device)))

    if params["restore"]:   
        aG.load_state_dict(torch.load(params["restore_path"] + params["gen_saved"], map_location=torch.device(device)))
        aD.load_state_dict(torch.load(params["restore_path"] + params["crit_saved"], map_location=torch.device(device)))
        
    else:
        aG.apply(weights_init)
        aD.apply(weights_init)

    if params["c0"]: 
        aE.apply(weights_init)
    elif params["c1"] :
        aE.load_state_dict(torch.load(params["calib_saved"], map_location=torch.device(device)))
    
    
    
    one = torch.tensor(1.0).to(device)
    mone = (one * -1).to(device)



    print('loading data...')
    paths_list = [
        '/beegfs/desy/user/eren/data_generator/pion/hcal_only/pion40part1.hdf5',
        '/beegfs/desy/user/eren/data_generator/pion/hcal_only/pion40part2.hdf5',
        '/beegfs/desy/user/eren/data_generator/pion/hcal_only/pion40part3.hdf5',
        '/beegfs/desy/user/eren/data_generator/pion/hcal_only/pion40part4.hdf5',
        '/beegfs/desy/user/eren/data_generator/pion/hcal_only/pion40part5.hdf5',
        '/beegfs/desy/user/eren/data_generator/pion/hcal_only/pion40part6.hdf5',
        '/beegfs/desy/user/eren/data_generator/pion/hcal_only/pion40part7.hdf5'
    ]


    train_data = PionsDataset(paths_list)
    dataloader = DataLoader(train_data, shuffle=True, batch_size=hyperp["batch_size"], num_workers=10)
    print('done')


    optimizer_g = torch.optim.Adam(aG.parameters(), lr=hyperp["L_gen"], betas=(0.5, 0.9))
    optimizer_d = torch.optim.Adam(aD.parameters(), lr=hyperp["L_crit"], betas=(0.5, 0.9))
    optimizer_e = torch.optim.SGD(aE.parameters(), lr=hyperp["L_calib"])
    #optimizer_p = torch.optim.Adam(aP.parameters(), lr=params["L_post"], betas=(0.5, 0.9))


    #scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=1, gamma=params["gamma_g"])
    #scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=1, gamma=params["gamma_crit"])
    #scheduler_e = optim.lr_scheduler.StepLR(optimizer_e, step_size=1, gamma=params["gamma_calib"])
   
    
    #writer = SummaryWriter()

    e_criterion = nn.L1Loss() # for energy regressor training

    dataiter = iter(dataloader)
    
    BATCH_SIZE = hyperp["batch_size"]
    LATENT = hyperp["z"]
    EXP = params["exp"]
    KAPPA = hyperp["kappa"]
    LAMBD = hyperp["lambda"]
    ## Post-Processing 
    LDP = hyperp["LDP"]
    wMMD = hyperp["wMMD"]
    wMSE = hyperp["wMSE"]

    ## IO paths
    OUTP = params['output_path']

    for iteration in range(1, 75000):
        
        #---------------------TRAIN D------------------------
        for p in aD.parameters():  # reset requires_grad
            p.requires_grad_(True)  # they are set to False below in training G
        
        for e in aE.parameters():  # reset requires_grad (constrainer)
            e.requires_grad_(True)  # they are set to False below in training G


        for i in range(hyperp["ncrit"]):
            
            aD.zero_grad()
            aE.zero_grad()
            
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, LATENT))    
            noise = torch.from_numpy(noise).float()
            noise = noise.view(-1, LATENT, 1, 1, 1)    #[BS, nz]  --> [Bs,nz,1,1,1] Needed for Generator
            noise = noise.to(device)
            
            batch = next(dataiter, None)

            if batch is None:
                dataiter = iter(dataloader)
                batch = dataiter.next()

            real_label = batch['energy'] ## energy label
            real_label = real_label.to(device)
           

            with torch.no_grad():
                noisev = noise  # totally freeze G, training D
            
            fake_data = aG(noisev, real_label).detach()
            

            real_data = batch['shower'] # 48x48x48 calo image
            real_data = real_data.to(device)
            real_data.requires_grad_(True)

        
            

            #### supervised-training for energy regressor!
            if params["train_calib"] :
                output = aE(real_data.float())
                e_loss = e_criterion(output, real_label.view(BATCH_SIZE, 1))
                e_loss.backward()
                optimizer_e.step()

            ######

            # train with real data
            
            disc_real = aD(real_data.float(), real_label.float())
        

            # train with fake data
            fake_data = fake_data.unsqueeze(1)  ## transform to [BS, 1, 30, 30, 30]
            disc_fake = aD(fake_data, real_label.float())

            
            # train with interpolated data
            gradient_penalty = calc_gradient_penalty(aD, real_data, fake_data, real_label, BATCH_SIZE, device, DIM=48)
            
            ## wasserstein-1 distace
            w_dist = torch.mean(disc_fake) - torch.mean(disc_real)
            # final disc cost
            disc_cost = torch.mean(disc_fake) - torch.mean(disc_real) + LAMBD * gradient_penalty
            disc_cost.backward()
            optimizer_d.step()
            
            #--------------Log to COMET ML ----------
            if i == hyperp["ncrit"]-1:
                experiment.log_metric("L_crit", disc_cost, step=iteration)
                experiment.log_metric("gradient_pen", gradient_penalty, step=iteration)
                experiment.log_metric("Wasserstein Dist", w_dist, step=iteration)
                if params["train_calib"]:
                    experiment.log_metric("L_const", e_loss, step=iteration)
        
        #---------------------TRAIN G------------------------
        for p in aD.parameters():
            p.requires_grad_(False)  # freeze D
        
        for c in aE.parameters():
            c.requires_grad_(False)  # freeze C

        gen_cost = None
        for i in range(hyperp["ngen"]):
            
            aG.zero_grad()
            
            
            noise = np.random.uniform(-1,1, (BATCH_SIZE, LATENT))
            noise = torch.from_numpy(noise).float()
            noise = noise.view(-1, LATENT, 1, 1, 1) #[BS, nz]  --> [Bs,nz,1,1,1] Needed for Generator
            noise = noise.to(device)


            batch = next(dataiter, None)

            if batch is None:
                dataiter = iter(dataloader)
                batch = dataiter.next()
            
            real_label = batch['energy'] ## energy label
            real_label = real_label.to(device)
            
            
            noise.requires_grad_(True)

            
            real_data = batch['shower'] # 48x48x48 calo image
            real_data = real_data.to(device)

            fake_data = aG(noise, real_label.float())
            fake_data = fake_data.unsqueeze(1)  ## transform to [BS, 1, 30, 30, 30]
            
            
            ## calculate loss function 
            gen_cost = aD(fake_data.float(), real_label.float())
            
            ## label conditioning
            output_g = aE(fake_data)
            output_r = aE(real_data.float())


            aux_fake = (output_g - real_label)**2
            aux_real = (output_r - real_label)**2
            
            aux_errG = torch.abs(aux_fake - aux_real)
            
            ## Total loss function for generator
            g_cost = -torch.mean(gen_cost) + KAPPA*torch.mean(aux_errG) 
            g_cost.backward()
            optimizer_g.step()

            #--------------Log to COMET ML ----------
            experiment.log_metric("L_Gen", g_cost, step=iteration)
            #experiment.log_metric("L_aux", aux_errG, step=iteration)

        #end = timer()
        #print(f'---train G elapsed time: {end - start}')

        if params["train_postP"]:
            #---------------------TRAIN P------------------------
            for p in aD.parameters():
                p.requires_grad_(False)  # freeze D
            
            for c in aG.parameters():
                c.requires_grad_(False)  # freeze G

            lossP = None
            for i in range(1):
                
                noise = np.random.uniform(-1,1, (BATCH_SIZE, LATENT))
                noise = torch.from_numpy(noise).float()
                noise = noise.view(-1, LATENT, 1, 1, 1) #[BS, nz]  --> [Bs,nz,1,1,1] Needed for Generator
                noise = noise.to(device)


                batch = next(dataiter, None)

                if batch is None:
                    dataiter = iter(dataloader)
                    batch = dataiter.next()

                real_label = batch[1] ## energy label
                real_label = real_label.unsqueeze(-1)  ## transform to [Bs, 1 ]
                real_label = real_label.to(device)
                real_label = real_label.view(-1, 1, 1, 1, 1)  #[BS,1] ---> [BS,1,1,1,1]  Needed for Generator
                noise.requires_grad_(True)

                real_data = batch[0] # 30x30x30 calo layers
                real_data = real_data.unsqueeze(1) #transform to [Bs, 1, 30, 30, 30 ]
                real_data = real_data.to(device)

                fake_data = aG(noise, real_label.float())
                            
                real_label = real_label.view(BATCH_SIZE, 1)   ## transform back : [BS,1,1,1,1]  -- > [BS,1]
                fake_data = fake_data.unsqueeze(1)  ## transform to [BS, 1, 30, 30, 30]
                
                ### first LossD_P
                fake_dataP = aP(fake_data.float(), real_label.float())
                lossD_P = aD(fake_dataP.float(), real_label.float())
                lossD_P = lossD_P.mean()

                ## lossFixP

                real_sorted = real_data.view(BATCH_SIZE, -1)
                fake_sorted = fake_dataP.view(BATCH_SIZE, -1)
                
                real_sorted, _ = torch.sort(real_sorted, dim=1, descending=True) #.view(900,1)
                fake_sorted, _ = torch.sort(fake_sorted, dim=1, descending=True) #.view(900,1)

                lossFixPp1 = mmd_hit_sortKernel(real_sorted.float(), fake_sorted, kernel_size=100, stride=50, cutoff=2000, alpha=200) 
                
                
                lossFixPp2 = F.mse_loss(fake_dataP.view(BATCH_SIZE, -1), 
                                        fake_data.detach().view(BATCH_SIZE, -1), reduction='mean')
                
                lossFixP = wMMD*lossFixPp1 + wMSE*lossFixPp2

                lossP = LDP*lossD_P - lossFixP

                lossP.backward(mone)            
                optimizer_p.step()



        
       

        #if params["train_postP"]:
        #    writer.add_scalar('data/lossD_P', lossD_P.mean(), iteration)
        #    writer.add_scalar('data/lossMSE', lossFixPp2.mean(), iteration)
        #    writer.add_scalar('data/lossMMD', lossFixPp1.mean(), iteration)
    

        if iteration % 1000==999 or iteration == 1 :
            print ('iteration: {}, critic loss: {}'.format(iteration, disc_cost.cpu().data.numpy()) )
            torch.save(aG.state_dict(), OUTP+'{0}/netG_itrs_{1}.pth'.format(EXP, iteration))
            torch.save(aD.state_dict(), OUTP+'{0}/netD_itrs_{1}.pth'.format(EXP, iteration))
            if params["train_calib"] :
                torch.save(aE.state_dict(), OUTP+'/{0}/netE_itrs_{1}.pth'.format(EXP, iteration))
            if params["train_postP"]:
                torch.save(aP.state_dict(), OUTP+'{0}/netP_itrs_{1}.pth'.format(EXP, iteration))
        
        #scheduler_d.step()
        #scheduler_g.step()
        #scheduler_e.step()

        

def main():
    
    default_params = {

        ## IO parameters
        "output_path" : '/beegfs/desy/user/eren/HCAL-showers/WGAN/output/',
        "exp" : 'wGANv1-bs100',                   ## where the models will be saved!
        "data_dim" : 3,
        ## optimizer parameters 
        "opt" : 'Adam',
        "gamma_g" : 1.0,                    ## not used at the moment 
        "gamma_crit" : 1.0,                 ## not used at the moment
        "gamma_calib" : 1.0,                ## not used at the moment
        ## checkpoint parameters
        "restore" : False,
        "restore_pp" : False,
        "restore_path" : '/*-path-to-checkpoint-folder*/',
        "restore_path_PP": '/*-path-checkpoint-folder-postprocessing*/',
        "gen_saved" : 'netG_itrs_XXXX.pth',
        "crit_saved" : 'netD_itrs_XXXX.pth',
        "calib_saved" : '/beegfs/desy/user/eren/HCAL-showers/WGAN/output/wGANv0/netE_itrs_1.pth',
        "post_saved" : 'netP_itrs_XXXX.pth',
        "c0" : True,                   ## randomly starts calibration networks parameters
        "c1" : False,                    ## starts from a saved model
        "train_calib": False,           ## you might want to turn off constrainer network training
        "train_postP": False,
        "seed": 32,


    }

    hyper_params = {
        ## general 
        "batch_size" : 100,
        "lambda" : 5,
        "kappa" : 0.0,
        "ncrit" : 10,
        "ngen" : 1,
        ## learning rate 
        "L_gen" : 1e-04,
        "L_crit" : 1e-05,
        "L_calib" : 1e-05,
        "L_post"  : 1e-07,
        ## model parameters
        "ngf" : 32,  
        "ndf" : 32,
        "z" : 100,
        ### hyper-parameters for post-processing
        "LDP" : 0.0,
        "wMMD" : 5.0,
        "wMSE" : 1.0,

    }

    train(default_params,hyper_params)


if __name__ == "__main__":
    main()


    

    
    

    















