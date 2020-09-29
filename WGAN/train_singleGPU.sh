#!/bin/bash

#SBATCH --partition=maxgpu
#SBATCH --nodes=1                                 # Number of nodes
#SBATCH --time=24:00:00     
#SBATCH --constraint="V100"
#SBATCH --chdir   /beegfs/desy/user/eren          # directory must already exist!
#SBATCH --job-name  wGANv0
#SBATCH --output    wGANv0-%N.out            # File to which STDOUT will be written
#SBATCH --error     wGANv0-%N.err            # File to which STDERR will be written
#SBATCH --wait-all-nodes=1
#SBATCH --mail-type END 

## go to the target directory
cd /beegfs/desy/user/eren/


## Setup tmp and cache directory of singularity
export SINGULARITY_TMPDIR=/beegfs/desy/user/eren/container/tmp/
export SINGULARITY_CACHEDIR=/beegfs/desy/user/eren/container/cache/

#name of the instance / experiment
INS=wGANv0

# necessary for output weights
mkdir -p /beegfs/desy/user/eren/HCAL-showers/WGAN/output/$INS

## start the container
singularity run -H $PWD --nv docker://engineren/pytorch:latest python HCAL-showers/WGAN/wGAN.py
