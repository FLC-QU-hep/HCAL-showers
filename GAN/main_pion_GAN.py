
import random
import torch
import sys
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.autograd.variable import Variable
import numpy as np
import models.GAN_pi as pion_GAN
from torch.utils.data import DataLoader
from models.dataUtils import PionsDataset
from apex import amp
from tqdm import tqdm
import time

# Set random seed for reproducibility
# manualSeed = random.randint(1, 10000) # use if you want new results
manualSeed = 2517
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Number of workers for dataloader
workers = 35
# Batch size during training
batch_size = 32
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Number of training epochs
num_epochs = 50
# Learning rate for optimizers
lr = 0.00001
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# Historical averaging rate
hist_rate = 3

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


PATH_save = '/beegfs/desy/user/akorol/trained_models/pion_gan/condition_pion_gan_{}.pth'
PATH_chechpoint = '/beegfs/desy/user/akorol/trained_models/pion_gan/pion_gan_20.pth'

def save(netG, netD, omtim_G, optim_D, epoch, loss, scores, path_to_save, time_stats):
    torch.save({
                'Generator': netG.state_dict(),
                'Discriminator': netD.state_dict(),
                'G_optimizer': omtim_G.state_dict(),
                'D_optimizer': optim_D.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'D_scores': scores,
                'time_stats': time_stats
                },
                path_to_save)

netD = pion_GAN.Discriminator(ngpu).to(device)
netG = pion_GAN.Generator(ngpu).to(device)
print(netG, netD)

# Apply the weights_init function to randomly initialize all weights
netD.apply(pion_GAN.weights_init)
netG.apply(pion_GAN.weights_init)

# Initialize BCELoss function
criterion = nn.BCEWithLogitsLoss()

# Optimizers
optimizer_G = optim.Adam(netG.parameters(), lr=lr*100, betas=(beta1, 0.999))
optimizer_D = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

# netD, optimizer_D = amp.initialize(netD, optimizer_D, opt_level="O1")
# netG, optimizer_G = amp.initialize(netG, optimizer_G, opt_level="O1")

if len(sys.argv) > 1:
    if sys.argv[1] == 'fromchp':
        print('loading from checkpoint...')
        checkpoint = torch.load(PATH_chechpoint)
        netG.load_state_dict(checkpoint['Generator'])
        netD.load_state_dict(checkpoint['Discriminator'])
        optimizer_G.load_state_dict(checkpoint['G_optimizer'])
        optimizer_D.load_state_dict(checkpoint['D_optimizer'])
        eph = checkpoint['epoch']
        chechoint_loss = checkpoint['loss']
        time_stats = checkpoint['time_stats']
        G_losses = chechoint_loss[0]
        D_losses = chechoint_loss[1]
        chechoint_scores = checkpoint['D_scores']
        D_scores_x = chechoint_scores[0]
        D_scores_z1 = chechoint_scores[1]
        D_scores_z2 = chechoint_scores[2]
        time_stats_epoch = time_stats[0]
        time_stats_calc = time_stats[1]
        print('done')
    else:
        print('Unexpected argument: ', sys.argv[1])
        exit()

else:
    print('start from scratch')
    eph = 0
    G_losses = np.array([])
    D_losses = np.array([])
    D_scores_x = np.array([])
    D_scores_z1 = np.array([])
    D_scores_z2 = np.array([])
    time_stats_calc = np.array([])
    time_stats_epoch = np.array([])



# Handle multi-gpu if desired
# if (device.type == 'cuda') and (ngpu > 1):
#     netD = nn.DataParallel(netD, list(range(ngpu)))
#     netG = nn.DataParallel(netG, list(range(ngpu)))
# else:
#     netD = nn.DataParallel(netD)
#     netG = nn.DataParallel(netG)
netG.train()
netD.train()



print('loading data...')
paths_list = ['/beegfs/desy/user/eren/data_generator/pion/hcal_only/pion40part1.hdf5',
              '/beegfs/desy/user/eren/data_generator/pion/hcal_only/pion40part2.hdf5',
              '/beegfs/desy/user/eren/data_generator/pion/hcal_only/pion40part3.hdf5',
              '/beegfs/desy/user/eren/data_generator/pion/hcal_only/pion40part4.hdf5',
              '/beegfs/desy/user/eren/data_generator/pion/hcal_only/pion40part5.hdf5',
              '/beegfs/desy/user/eren/data_generator/pion/hcal_only/pion40part6.hdf5',
              '/beegfs/desy/user/eren/data_generator/pion/hcal_only/pion40part7.hdf5'
              ]

train_data = PionsDataset(paths_list)
dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=workers)
print('done')

print('Start training loop')
for epoch in tqdm(range(num_epochs)):
    epoch += eph + 1

    epoch_time = time.time()
    calculation_time = 0
    for batch in tqdm(dataloader):
        calc_time = time.time()
        netG.train()
        btch_sz = len(batch['shower'])
        real_showers = Variable(batch['shower']).float().to(device)
#         real_showers[real_showers<0.3] = 0
#         real_energys = Variable(batch['energy']*0).float().to(device)
        real_free_path = batch['free_path'].float().to(device).reshape(btch_sz, 1, 1, 1, 1)

        # Adversarial ground truths
        valid_label = Variable(FloatTensor(btch_sz, 1).fill_(1.0), requires_grad=False)
        fake_label = Variable(FloatTensor(btch_sz, 1).fill_(0.0), requires_grad=False)


        ######################################################
        # Train Discriminator
        ######################################################
        netD.zero_grad()

        # Forward pass real batch through Disctiminator
        output = netD(real_showers, real_free_path)

        # Calculate loss on all-real batch
        errD_real = criterion(output, valid_label)

        # Calculate gradients for D in backward pass
        errD_real.backward()
#         with amp.scale_loss(errD_real, optimizer_D) as scaled_errD_real:
#             errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
#         noise = torch.randn(btch_sz, nz, 1, 1, 1, device=device)
        noise = torch.FloatTensor(btch_sz, 100, 1, 1, 1).uniform_(-1, 1)
        # labels for Generator
        gen_free_path = torch.Tensor(btch_sz, 1, 1, 1, 1).uniform_(1, 49).type(torch.int)
        gen_free_path = gen_free_path.to(device).float()
#         gen_labels = np.random.uniform(10, 100, btch_sz)
#         gen_labels = Variable(FloatTensor(gen_labels))
#         gen_labels = gen_labels.view(btch_sz, 1, 1, 1, 1)*0

        # Generate fake image batch with G
        fake_shower = netG(noise.to(device), gen_free_path)

        # Classify all fake batch with D
        output = netD(fake_shower, gen_free_path)

        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, fake_label)

        # Calculate the gradients for this batch
        errD_fake.backward(retain_graph=True)
#         with amp.scale_loss(errD_fake, optimizer_D) as scaled_errD_fake:
#             errD_fake.backward(retain_graph=True)
        D_G_z1 = output.mean().item()

        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake

        # Update D
        optimizer_D.step()


        ######################################################
        # Train Generator
        ######################################################
        netG.zero_grad()

        # Forward pass of all-fake batch through D
        gen_free_path = 48 - gen_free_path + 1
        output = netD(fake_shower, gen_free_path)

        # Calculate G's loss based on this output
        errG = criterion(output, valid_label)

        # Calculate gradients for G
        errG.backward(retain_graph=True)
#         with amp.scale_loss(errG, optimizer_G) as scaled_errG:
#             errG.backward(retain_graph=True)
        D_G_z2 = output.mean().item()

        # Update G
        optimizer_G.step()

        # Output training stats
        G_losses = np.append(G_losses, errG.item())
        D_losses = np.append(D_losses, errD.item())
        D_scores_x = np.append(D_scores_x, D_x)
        D_scores_z1 = np.append(D_scores_z1, D_G_z1)
        D_scores_z2 = np.append(D_scores_z2, D_G_z2)

        calculation_time += time.time() - calc_time

    epoch_t = time.time()-epoch_time
    print('[%d/%d], Loss_D: %.4f, Loss_G: %.4f, D(x): %.4f,  D(G(z)): %.4f/%.4f,\n\t epoch time: %.2f, calculation time: %.2f'
          % (epoch, num_epochs, errD.item(), errG.item(),
             D_x, D_G_z1, D_G_z2, epoch_t, calculation_time))

    time_stats_epoch = np.append(time_stats_epoch, epoch_t)
    time_stats_calc = np.append(time_stats_calc, calculation_time)

    if epoch%5 == 0:
        time_stats = np.array([time_stats_epoch, time_stats_calc])
        loss =  np.array([G_losses, D_losses])
        D_scores = np.array([D_scores_x, D_scores_z1, D_scores_z2])
        save(netG=netG, netD=netD, omtim_G=optimizer_G, optim_D=optimizer_D,
             epoch=epoch, loss=loss, scores=D_scores, time_stats = time_stats,
             path_to_save=PATH_save.format(epoch))
