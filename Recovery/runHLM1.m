% runHLM provides inputs to setHLM for a given run on the cluster

% It provides the following inputs when calling setHLM:

% mode        - 1=simulate CPT (inverse-S) choices,
%               2=simulate LML choices, 
%               3=simulate CPT (regular-S) choices,
%               4=model recovery for CPT (inverse-S) choices, 
%               5=model recovery for LML choices,
%               6=model recovery for CPT (regular-S) choices, 
%               7=parameter revovery CPT (inverse-S), 
%               8=parameter 'recovery' LML,
%               9=parameter revovery CPT (regular-S). 
% whichJAGS   - which copy of matjags to run on. this allows parallel jobs 
%               to run as long as they use different matjags
% whichQuals  - sets the order of qualities to run
% nGambles    - Number of gambles to run over
% nAgents     - number of synthetic agents 

% The idea is that this is written into by the user, then called by a
% cluster job via the terminal:

%% Add to path
%cd .. ;%move to base directory 
addpath(genpath(pwd));%adds base directory and subfolders to path, important for running shell scripts from terminal

%% Specify variables
for i =1:9
mode=i;
whichJAGS=1;
whichQuals=4;
nGambles=24;
doParallel=1;
nAgents=10;


%% Call setHLM
setHLM(mode,whichJAGS,whichQuals,doParallel,nGambles,nAgents)
end