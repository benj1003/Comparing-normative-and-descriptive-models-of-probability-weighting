% runHLM provides inputs to setHLM for a given run on the cluster

% It provides the following inputs when calling setHLM:

% mode        - 1=simulate CPT choices,2=simulate LML choices, 
%               3=model recovery for CPT choices, 4=model recovery for LML choices
%               5=parameter revovery CPT, 6=parameter 'recovery' LML
% whichJAGS   - which copy of matjags to run on. this allows parallel jobs to run as long as they use different matjags
% whichQuals  - sets the order of qualities to run
% nGambles    - Number of gambles to run over
% nAgents     - number of synthetic agents 

% The idea is that this is written into by the user, then called by a
% cluster job via the terminal:

%% Add to path
%cd .. ;%move to base directory 
addpath(genpath(pwd));%adds base directory and subfolders to path, important for running shell scripts from terminal

%% Specify variables
mode=3;
whichJAGS=1;
whichQuals=1;
nGambles=1;
doParallel=0;
nAgents=2 ;


%% Call setHLM
setHLM(mode,whichJAGS,whichQuals,doParallel,nGambles,nAgents)