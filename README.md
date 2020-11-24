# Master-thesis: Normative accounts of Probability Weighting in Decision Science

This reposetory contains all code needed to replicate the work done in the final project done to fulfill the requirements to obtain a masters degree in Mathematical Modelling and Computation from the Technical University of Denmark (DTU) by stud. polyt Benjamin Skjold Frederiksen. 

## Data
The reposetory contains all data needed to replicate all findings. However, it also contains the code that was used to generate the pseudo-random data, such that the full process can be replicated. 

## Platform/Environment
The raw data (the underlying gambles) is in .csv format. A python script (Python 3.7.1) reads this and generate the trials for each gamble (converting from decision from description to decisions from experience).

Most subsequent analyses have been performed with, or called from Matlab. The Hierarchical Bayesian modelling has been run with JAGS (via matjags, which allows running JAGS via matlab code).

## Folder summary

- Figures: Contains all figures used within the project (both final paper and various presentations)
- Generating gambles: Contains the .cvs file with all underlying gambles, as well as the Python script used to convert each gambles into a series of trials
- Presentations: Contains slides used in presentations regarding the project
- Recovery: Contains all files used to run simulations (including simulation choices, model recovery and parameter recovery)

## Setting up JAGS
JAGS is run via matjags - see https://github.com/msteyvers/matjags for detailed description of instalation.

## Detailed Setup Instructions - running JAGS
Runnning a model starts with setting up the script runHLM#.m where # indicates the version number. There are multiple versions according to the aims of the model, which can vary by the aim (simulating choices, model recovery or parameter recovery)

To call a given runHLM# you need to run the job runHLM_job#.sh in the "jobs" directory. #NOTE!!!

Each version of runHLM contains different specifications for running a JAGS model, pertaining to the following key variables: 
- mode: 1=simulate CPT choices,2=simulate LML choices, 3=model recovery for CPT choices, 4=model recovery for LML choices, 5=parameter revovery CPT, 6=parameter 'recovery' LML
- whichJAGS: which copy of matjags to run on. this allows parallel jobs to run as long as they use different matjags
- whichQuals: sets the order of qualities to run
- nGambles: number of gambles to run over
- nAgents: number of synthetic agents 

runHLM# runHLM#.m (in folder 'jobs') calls setHLM.m (in "Recovery" folder) which contains preset information about what the different quality levels mean in terms of burn-in, number of samples etc.

setHLM setHLM then calls computeHLM which is the main script for computing the hieararchical bayesian models. This script processes the data, and principally calls matjags to run the model. This is done by calling the different versions of the "JAGS_..." scripts using JAGS code, to be found in the folder "Recovery/JAGS"

computeHLM is a general script for running several types of hierarchical bayesian model via JAGS. It can run hiearchical latent mixture models in which different utility models can be compared via inference on the model indicator variables, and it can run without latent mixtures of models for instance in order to estimate parameters of a given utility model.


