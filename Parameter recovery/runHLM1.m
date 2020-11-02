% runHLM provides inputs to setHLM for a given run on the cluster

% It provides the following inputs when calling setHLM:

% runModelNum - which models to run
% mode        - simulating data or recovering parameters
% whichJAGS   - which copy of matjags to run on. this allows parallel jobs to run as long as they use different matjags
% whichQuals  - sets the order of qualities to run
% whichGamble - sets which underlying gamble is considered
% permuted    - set if the model should be run on sequence where the trials are permuted among samples

% The idea is that this is written into by the user, then called by a
% cluster job via the terminal:

%% Add to path
%cd .. ;%move to base directory 
%addpath(genpath(pwd));%adds base directory and subfolders to path, important for running shell scripts from terminal

%% Specify variables
runModelNum=1;
Mode=1;
whichJAGS=1;
whichQuals=1;
whichGamble=0;
permuted='False'; %True not implemented (yet)
doParallel=0;
nAgents=100;


%% Call setHLM
setHLM(runModelNum,Mode,whichJAGS,whichQuals,doParallel,whichGamble,permuted,nAgents)