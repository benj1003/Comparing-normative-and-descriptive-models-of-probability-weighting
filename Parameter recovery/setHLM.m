function setHLM(Mode,whichJAGS,whichQuals,doParallel,nGambles,nAgents)
%% setHLM
% setHLM sets up multiple HLM models to run sequentially according to inputs
% This function takes the following inputs:
% mode        - 1=simulate CPT choices,2=simulate LML choices, 
%               3=model recovery, 4=parameter revovery CPT,
%               5=parameter 'recovery' LML
% whichJAGS   - which copy of matjags to run on. this allows parallel jobs to run as long as they use different matjags
% whichQuals  - sets the order of qualities to run
% nGambles    - Number of gambles to run over
% nAgents     - number of synthetic agents 
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
    computeHLM(nBurnin(whichQuals(i)),nSamples(whichQuals(i)),nThin,nChains(whichQuals(i)),whichJAGS,Mode,doParallel,nGambles,nAgents)
end