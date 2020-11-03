function setHLM(runModelNum,Mode,whichJAGS,whichQuals,doParallel,nGambles,permuted,nAgents)
%% setHLM
% setHLM sets up multiple HLM models to run sequentially according to inputs
% This function takes the following inputs:
% runModelNum - which models to run
% mode        - simulating data or recovering parameters
% whichJAGS   - which copy of matjags to run on. this allows parallel jobs to run as long as they use different matjags
% whichQuals  - sets the order of qualities to run
% whichGamble - sets which underlying gamble is considered
% permuted    - set if the model should be run on sequence where the trials are permuted among samples
% 
% There are three qualities for several variables, each selected by whichQuals
% qualities  are 'bronze','silver','gold'
% gold is highest quality but takes longest, bronzest lowest but fastest
% etc.

%% Specifies qualities to be selected from
numRuns      = length(whichQuals);%how many separate instances of an MCMC to run
nBurnin      = [1e2,1e3,1e4,2e4,4e4];%from 100 to 40k
nSamples     = [5e1,5e2,5e3,1e4,2e4];%from 50 to 20k
nChains      = [4,4,4,4,4];%
nThin        = 1;%thinnning factor, 1 = no thinning, 2=every 2nd etc.
%% Runs HLMs sequentiallt
for i=1:numRuns
    computeHLM(runModelNum,nBurnin(whichQuals(i)),nSamples(whichQuals(i)),nThin,nChains(whichQuals(i)),whichJAGS,Mode,doParallel,nGambles,permuted,nAgents)
end