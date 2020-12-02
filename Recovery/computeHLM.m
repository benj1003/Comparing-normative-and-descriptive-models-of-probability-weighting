function computeHLM(nBurnin,nSamples,nThin,nChains,whichJAGS,mode,doParallel,nGambles,nAgents)

%% Hiercharchical Latent Mixture (HLM) model
% This is a general script for running several types of hierarchical
% bayesian model via JAGS. It can run hiearchical latent mixture models in
% which different utility models can be compared via inference on the model
% indicator variables, and it can run without latent mixtures of models for
% instance in order to estimate parameters of a given utility model. It
% takes as input the following:

% nBurnin - a number which specifies how many burn in samples to run.
% nThin - a number specifying the thinnning number
% nChains - number of chains
% mode - a numbber identifying which part of the analysis to run


%% Set paths
[startDir,~] = fileparts(mfilename('fullpath'));%specify your starting directory here (where this script runs from)
cd(startDir);%move to starting directory
jagsDir=[startDir,'/JAGS'];
addpath(fullfile(pwd,'/matjags'));%set path to include matjags folder
addpath(fullfile(pwd,'/data'));%set path to include data folder
addpath(fullfile(pwd,'/samples_stats'));%set path to include samples_stats folder

%% Set model name
modelName = 'JAGS'; %same model used for all modes

%% Set bounds of hyperpriors

switch mode
    case 1
        %beta - prior on log since cannot be less than 0; note same bounds used for independent priors on all models
        muLogBetaL=-2.3;muLogBetaU=3.4;muLogBetaM=(muLogBetaL+muLogBetaU)/2; %bounds on mean of distribution log beta
        sigmaLogBetaL=0.01;sigmaLogBetaU=sqrt(((muLogBetaU-muLogBetaL)^2)/12);sigmaLogBetaM=(sigmaLogBetaL+sigmaLogBetaU)/2;%bounds on the std of distribution of log beta
        
        %Alpha - prior on log since cannot be less than 0; note same bounds used for independent priors on all models
        muLogAlphaL=-0.01;muLogAlphaU=0.01;muLogAlphaM=(muLogAlphaL+muLogAlphaU)/2;%bounds on mean of distribution of log Alpha
        sigmaLogAlphaL=0.99;sigmaLogAlphaU=1.01;sigmaLogAlphaM=(sigmaLogAlphaL+sigmaLogAlphaU)/2;
        %sqrt(((muLogAlphaU-muLogAlphaL)^2)/12)-0.01;sigmaLogAlphaU=sqrt(((muLogAlphaU-muLogAlphaL)^2)/12)+0.01;sigmaLogAlphaM=(sigmaLogAlphaL+sigmaLogAlphaU)/2; %bounds on std of distribution of log Alpha

        %Delta - prior on log since cannot be less than 0
        muLogDeltaL=-0.21;muLogDeltaU=-0.19;muLogDeltaM=(muLogDeltaL+muLogDeltaU)/2;%bounds on mean of distribution of log Delta
        sigmaLogDeltaL=1.5;sigmaLogDeltaU=1.6;sigmaLogDeltaM=(sigmaLogDeltaL+sigmaLogDeltaU)/2; %bounds on std of distribution of log Delta

        %Gamma - prior on log since cannot be less than 0
        muLogGammaL=-0.81;muLogGammaU=-0.79;muLogGammaM=(muLogGammaL+muLogGammaU)/2;%bounds on mean of distribution of log Gamma
        sigmaLogGammaL=1.5;sigmaLogGammaU=1.6;sigmaLogGammaM=(sigmaLogGammaL+sigmaLogGammaU)/2; %bounds on std of distribution of log Gamma

    case 2
        %beta - prior on log since cannot be less than 0; note same bounds used for independent priors on all models
        muLogBetaL=-2.3;muLogBetaU=3.4;muLogBetaM=(muLogBetaL+muLogBetaU)/2; %bounds on mean of distribution log beta
        sigmaLogBetaL=0.01;sigmaLogBetaU=sqrt(((muLogBetaU-muLogBetaL)^2)/12);sigmaLogBetaM=(sigmaLogBetaL+sigmaLogBetaU)/2;%bounds on the std of distribution of log beta
        
        %Alpha - prior on log since cannot be less than 0; note same bounds used for independent priors on all models
        muLogAlphaL=-0.01;muLogAlphaU=0.01;muLogAlphaM=(muLogAlphaL+muLogAlphaU)/2;%bounds on mean of distribution of log Alpha
        sigmaLogAlphaL=0.99;sigmaLogAlphaU=1.01;sigmaLogAlphaM=(sigmaLogAlphaL+sigmaLogAlphaU)/2; %bounds on std of distribution of log Alpha
        %sqrt(((muLogAlphaU-muLogAlphaL)^2)/12)-0.01;sigmaLogAlphaU=sqrt(((muLogAlphaU-muLogAlphaL)^2)/12)+0.01;sigmaLogAlphaM=(sigmaLogAlphaL+sigmaLogAlphaU)/2; %bounds on std of distribution of log Alpha

        %Delta - prior on log since cannot be less than 0
        muLogDeltaL=-0.21;muLogDeltaU=-0.19;muLogDeltaM=(muLogDeltaL+muLogDeltaU)/2;%bounds on mean of distribution of log Delta
        sigmaLogDeltaL=1.5;1.6;sigmaLogDeltaM=(sigmaLogDeltaL+sigmaLogDeltaU)/2; %bounds on std of distribution of log Delta

        %Gamma - prior on log since cannot be less than 0
        muLogGammaL=-0.81;muLogGammaU=-0.79;muLogGammaM=(muLogGammaL+muLogGammaU)/2;%bounds on mean of distribution of log Gamma
        sigmaLogGammaL=1.5;sigmaLogGammaU=1.6;sigmaLogGammaM=(sigmaLogGammaL+sigmaLogGammaU)/2; %bounds on std of distribution of log Gamma
       
    case 3
        %beta - prior on log since cannot be less than 0; note same bounds used for independent priors on all models
        muLogBetaL=-2.3;muLogBetaU=3.4;muLogBetaM=(muLogBetaL+muLogBetaU)/2; %bounds on mean of distribution log beta
        sigmaLogBetaL=0.01;sigmaLogBetaU=sqrt(((muLogBetaU-muLogBetaL)^2)/12);sigmaLogBetaM=(sigmaLogBetaL+sigmaLogBetaU)/2;%bounds on the std of distribution of log beta
        
        %Alpha - prior on log since cannot be less than 0; note same bounds used for independent priors on all models
        muLogAlphaL=-2.3;muLogAlphaU=0;muLogAlphaM=(muLogAlphaL+muLogAlphaU)/2;%bounds on mean of distribution of log Alpha
        sigmaLogAlphaL=0.01;sigmaLogAlphaU=sqrt(((muLogAlphaU-muLogAlphaL)^2)/12);sigmaLogAlphaM=(sigmaLogAlphaL+sigmaLogAlphaU)/2; %bounds on std of distribution of log Alpha

        %Delta - prior on log since cannot be less than 0
        muLogDeltaL=-2.3;muLogDeltaU=0;muLogDeltaM=(muLogDeltaL+muLogDeltaU)/2;%bounds on mean of distribution of log Delta
        sigmaLogDeltaL=0.01;sigmaLogDeltaU=sqrt(((muLogDeltaU-muLogDeltaL)^2)/12);sigmaLogDeltaM=(sigmaLogDeltaL+sigmaLogDeltaU)/2; %bounds on std of distribution of log Delta

        %Gamma - prior on log since cannot be less than 0
        muLogGammaL=-2.3;muLogGammaU=0;muLogGammaM=(muLogGammaL+muLogGammaU)/2;%bounds on mean of distribution of log Gamma
        sigmaLogGammaL=0.01;sigmaLogGammaU=sqrt(((muLogGammaU-muLogGammaL)^2)/12);sigmaLogGammaM=(sigmaLogGammaL+sigmaLogGammaU)/2; %bounds on std of distribution of log Gamma
    
    case 4
        %beta - prior on log since cannot be less than 0; note same bounds used for independent priors on all models
        muLogBetaL=-2.3;muLogBetaU=3.4;muLogBetaM=(muLogBetaL+muLogBetaU)/2; %bounds on mean of distribution log beta
        sigmaLogBetaL=0.01;sigmaLogBetaU=sqrt(((muLogBetaU-muLogBetaL)^2)/12);sigmaLogBetaM=(sigmaLogBetaL+sigmaLogBetaU)/2;%bounds on the std of distribution of log beta
        
        %Alpha - prior on log since cannot be less than 0; note same bounds used for independent priors on all models
        muLogAlphaL=-2.3;muLogAlphaU=0;muLogAlphaM=(muLogAlphaL+muLogAlphaU)/2;%bounds on mean of distribution of log Alpha
        sigmaLogAlphaL=0.01;sigmaLogAlphaU=sqrt(((muLogAlphaU-muLogAlphaL)^2)/12);sigmaLogAlphaM=(sigmaLogAlphaL+sigmaLogAlphaU)/2; %bounds on std of distribution of log Alpha

        %Delta - prior on log since cannot be less than 0
        muLogDeltaL=-2.3;muLogDeltaU=0;muLogDeltaM=(muLogDeltaL+muLogDeltaU)/2;%bounds on mean of distribution of log Delta
        sigmaLogDeltaL=0.01;sigmaLogDeltaU=sqrt(((muLogDeltaU-muLogDeltaL)^2)/12);sigmaLogDeltaM=(sigmaLogDeltaL+sigmaLogDeltaU)/2; %bounds on std of distribution of log Delta

        %Gamma - prior on log since cannot be less than 0
        muLogGammaL=-2.3;muLogGammaU=0;muLogGammaM=(muLogGammaL+muLogGammaU)/2;%bounds on mean of distribution of log Gamma
        sigmaLogGammaL=0.01;sigmaLogGammaU=sqrt(((muLogGammaU-muLogGammaL)^2)/12);sigmaLogGammaM=(sigmaLogGammaL+sigmaLogGammaU)/2; %bounds on std of distribution of log Gamma
    
    case 5
        %beta - prior on log since cannot be less than 0; note same bounds used for independent priors on all models
        muLogBetaL=-2.3;muLogBetaU=3.4;muLogBetaM=(muLogBetaL+muLogBetaU)/2; %bounds on mean of distribution log beta
        sigmaLogBetaL=0.01;sigmaLogBetaU=sqrt(((muLogBetaU-muLogBetaL)^2)/12);sigmaLogBetaM=(sigmaLogBetaL+sigmaLogBetaU)/2;%bounds on the std of distribution of log beta
        
        %Alpha - prior on log since cannot be less than 0; note same bounds used for independent priors on all models
        muLogAlphaL=-2.3;muLogAlphaU=0;muLogAlphaM=(muLogAlphaL+muLogAlphaU)/2;%bounds on mean of distribution of log Alpha
        sigmaLogAlphaL=0.01;sigmaLogAlphaU=sqrt(((muLogAlphaU-muLogAlphaL)^2)/12);sigmaLogAlphaM=(sigmaLogAlphaL+sigmaLogAlphaU)/2; %bounds on std of distribution of log Alpha

        %Delta - prior on log since cannot be less than 0
        muLogDeltaL=-2.3;muLogDeltaU=0;muLogDeltaM=(muLogDeltaL+muLogDeltaU)/2;%bounds on mean of distribution of log Delta
        sigmaLogDeltaL=0.01;sigmaLogDeltaU=sqrt(((muLogDeltaU-muLogDeltaL)^2)/12);sigmaLogDeltaM=(sigmaLogDeltaL+sigmaLogDeltaU)/2; %bounds on std of distribution of log Delta

        %Gamma - prior on log since cannot be less than 0
        muLogGammaL=-2.3;muLogGammaU=0;muLogGammaM=(muLogGammaL+muLogGammaU)/2;%bounds on mean of distribution of log Gamma
        sigmaLogGammaL=0.01;sigmaLogGammaU=sqrt(((muLogGammaU-muLogGammaL)^2)/12);sigmaLogGammaM=(sigmaLogGammaL+sigmaLogGammaU)/2; %bounds on std of distribution of log Gamma
    
    case 6
        %beta - prior on log since cannot be less than 0; note same bounds used for independent priors on all models
        muLogBetaL=-2.3;muLogBetaU=3.4;muLogBetaM=(muLogBetaL+muLogBetaU)/2; %bounds on mean of distribution log beta
        sigmaLogBetaL=0.01;sigmaLogBetaU=sqrt(((muLogBetaU-muLogBetaL)^2)/12);sigmaLogBetaM=(sigmaLogBetaL+sigmaLogBetaU)/2;%bounds on the std of distribution of log beta
        
        %Alpha - prior on log since cannot be less than 0; note same bounds used for independent priors on all models
        muLogAlphaL=-2.3;muLogAlphaU=0;muLogAlphaM=(muLogAlphaL+muLogAlphaU)/2;%bounds on mean of distribution of log Alpha
        sigmaLogAlphaL=0.01;sigmaLogAlphaU=sqrt(((muLogAlphaU-muLogAlphaL)^2)/12);sigmaLogAlphaM=(sigmaLogAlphaL+sigmaLogAlphaU)/2; %bounds on std of distribution of log Alpha

        %Delta - prior on log since cannot be less than 0
        muLogDeltaL=-2.3;muLogDeltaU=0;muLogDeltaM=(muLogDeltaL+muLogDeltaU)/2;%bounds on mean of distribution of log Delta
        sigmaLogDeltaL=0.01;sigmaLogDeltaU=sqrt(((muLogDeltaU-muLogDeltaL)^2)/12);sigmaLogDeltaM=(sigmaLogDeltaL+sigmaLogDeltaU)/2; %bounds on std of distribution of log Delta

        %Gamma - prior on log since cannot be less than 0
        muLogGammaL=-2.3;muLogGammaU=0;muLogGammaM=(muLogGammaL+muLogGammaU)/2;%bounds on mean of distribution of log Gamma
        sigmaLogGammaL=0.01;sigmaLogGammaU=sqrt(((muLogGammaU-muLogGammaL)^2)/12);sigmaLogGammaM=(sigmaLogGammaL+sigmaLogGammaU)/2; %bounds on std of distribution of log Gamma

end

%% Choose & load data

for g = 1:nGambles %wrapped around the rest as JAGS cannot handle higher than 3 dimensions (until line 458)
switch mode
    case 1 %simulate choices with CPT
        dataSource = 'all_gambles';
        outputName = 'Choices_simulated_from_CPT'; priorName='';
        pz=[1,0,1,0,1,0,1,0];
        nChunks = 1;
        nSamples = 1;
        nChains = 1;
        
    case 2 %simulate choices with LML
        dataSource = 'all_gambles';
        outputName = 'Choices_simulated_from_LML'; priorName='';
        pz=[0,1,0,1,0,1,0,1];
        nChunks = 1;
        nSamples = 1;
        nChains = 1;
        
    case 3 %Model comparison on data from CPT
        dataSource = sprintf('Choices_simulated_from_CPT_Gamble_%.0f',g);
        outputName = 'model_comparison_CPT'; priorName='flat prior';
        pz=repmat(1/8,1,8);
        nChunks = 1;

    case 4 %Model comparison on data from LML
        dataSource = sprintf('Choices_simulated_from_LML_Gamble_%.0f',g);
        outputName = 'model_comparison_LML'; priorName='flat prior';
        pz=repmat(1/8,1,8);
        nChunks = 1;
        
    case 5 %parameter recovery for CPT data
        dataSource = sprintf('Choices_simulated_from_CPT_Gamble_%.0f',g);
        outputName = 'parameter_recovery_CPT'; priorName='';
        pz=[1,0,1,0,1,0,1,0];
        nChunks = 5; %to examine changes over time
        
    case 6 %parameter recovery for LML data
        dataSource = sprintf('Choices_simulated_from_LML_Gamble_%.0f',g);
        outputName = 'parameter_recovery_LML'; priorName='';
        pz=[1,0,1,0,1,0,1,0];
        nChunks = 5; %to examine changes over time
end

load(dataSource)

%% Set key variables
nTrials=1000;

chunkLength=nTrials/nChunks;
doDIC=0;%compute Deviance information criteria? This is the hierarchical equivalent of an AIC, the lower the better

%% Print information for user
disp('**************');
disp(['running model#_',modelName,':'])
disp(['_started:_',datestr(clock)])
disp(['MCMC number_',num2str(whichJAGS)])
disp(['running on_',dataSource])
disp(['with_',priorName])
disp('**************');

%% Initialise matrices
%initialise matrices with nan values of size subjects x conditions x trials
choice = nan(nAgents,nChunks,chunkLength); %initialise choice data matrix 
dx1 = nan(nAgents,nChunks,chunkLength); dx2 = dx1; dx3 = dx1; dx4=dx1;%initialise changes in wealth
p_a1  = nan(nAgents,nChunks,chunkLength); p_a2 = p_a1; p_b1 = p_a1; p_b2 = p_a1; %initialise channges in 'probabilities'

%% Compile choice & gamble data
%split into chunks for parameter recovery (mode 5 and 6)

switch mode
    case 1
        for i = 1:nAgents
            for c = 1:nChunks %nChunks = 1
                trialInds=1:length(Data{g,1}.Choice);%generate indices for each trial
                choice(i,c,trialInds)=Data{g,i}.Choice(trialInds);%assign to temporary variables

                dx1(i,c,trialInds)=Data{g,i}.maxA(trialInds);%assign changes in wealth dx for outcome 1 (note same amount for all trials)
                dx2(i,c,trialInds)=Data{g,i}.minA(trialInds);%same for outcome 2 etc.
                dx3(i,c,trialInds)=Data{g,i}.maxB(trialInds);
                dx4(i,c,trialInds)=Data{g,i}.minB(trialInds);

                p_a1(i,c,trialInds)=Data{g,i}.p_maxA(trialInds);%assign changes in 'probability' for outcome 1
                p_b1(i,c,trialInds)=Data{g,i}.p_maxB(trialInds);
            end
        end    
    
    case 2
        for i = 1:nAgents
            for c = 1:nChunks %nChunks = 1
                trialInds=1:length(Data{g,1}.Choice);%generate indices for each trial
                choice(i,c,trialInds)=Data{g,i}.Choice(trialInds);%assign to temporary variables

                dx1(i,c,trialInds)=Data{g,i}.maxA(trialInds);%assign changes in wealth dx for outcome 1 (note same amount for all trials)
                dx2(i,c,trialInds)=Data{g,i}.minA(trialInds);%same for outcome 2 etc.
                dx3(i,c,trialInds)=Data{g,i}.maxB(trialInds);
                dx4(i,c,trialInds)=Data{g,i}.minB(trialInds);

                p_a1(i,c,trialInds)=Data{g,i}.p_maxA(trialInds);%assign changes in 'probability' for outcome 1
                p_b1(i,c,trialInds)=Data{g,i}.p_maxB(trialInds);
            end
        end
        
    case 3
        for i = 1:nAgents
            trialInds=1:length(samples.y(1,1,1,1,:));%generate indices for each trial
            for c = 1:nChunks %nChunks = 1
                choice(i,c,trialInds)=samples.y(1,1,i,c,trialInds);%assign to temporary variables

                dx1(i,c,trialInds)=samples.dx1(1,1,i,c,trialInds);%assign changes in wealth dx for outcome 1 (note same amount for all trials)
                dx2(i,c,trialInds)=samples.dx2(1,1,i,c,trialInds);%same for outcome 2 etc.
                dx3(i,c,trialInds)=samples.dx3(1,1,i,c,trialInds);
                dx4(i,c,trialInds)=samples.dx4(1,1,i,c,trialInds);

                p_a1(i,c,trialInds)=samples.pa1(1,1,i,c,trialInds);%assign changes in 'probability' for outcome 1
                p_b1(i,c,trialInds)=samples.pb1(1,1,i,c,trialInds);
            end 
        end

    case 4
        for i = 1:nAgents
            trialInds=1:length(samples.y(1,1,1,1,:));%generate indices for each trial
            for c = 1:nChunks %nChunks = 1
                choice(i,c,trialInds)=samples.y(1,1,i,c,trialInds);%assign to temporary variables

                dx1(i,c,trialInds)=samples.dx1(1,1,i,c,trialInds);%assign changes in wealth dx for outcome 1 (note same amount for all trials)
                dx2(i,c,trialInds)=samples.dx2(1,1,i,c,trialInds);%same for outcome 2 etc.
                dx3(i,c,trialInds)=samples.dx3(1,1,i,c,trialInds);
                dx4(i,c,trialInds)=samples.dx4(1,1,i,c,trialInds);

                p_a1(i,c,trialInds)=samples.pa1(1,1,i,c,trialInds);%assign changes in 'probability' for outcome 1
                p_b1(i,c,trialInds)=samples.pb1(1,1,i,c,trialInds);
            end 
        end
        
    case 5
        for i = 1:nAgents
            trialInds_all=1:length(samples.y(1,1,1,1,:));%generate indices for each trial
            for c = 1:nChunks
                chunkInds=1:(length(samples.y(1,1,1,1,:))/nChunks);
                trialInds = trialInds_all(c*chunkLength-(chunkLength-1):c*chunkLength);

                choice(i,c,chunkInds)=samples.y(1,1,i,1,trialInds);%assign to temporary variables

                dx1(i,c,chunkInds)=samples.dx1(1,1,i,1,trialInds);%assign changes in wealth dx for outcome 1 (note same amount for all trials)
                dx2(i,c,chunkInds)=samples.dx2(1,1,i,1,trialInds);%same for outcome 2 etc.
                dx3(i,c,chunkInds)=samples.dx3(1,1,i,1,trialInds);
                dx4(i,c,chunkInds)=samples.dx4(1,1,i,1,trialInds);

                p_a1(i,c,chunkInds)=samples.pa1(1,1,i,1,trialInds);%assign changes in 'probability' for outcome 1
                p_b1(i,c,chunkInds)=samples.pb1(1,1,i,1,trialInds);
            end 
        end 
        
    case 6
        for i = 1:nAgents
            trialInds_all=1:length(samples.y(1,1,1,1,:));%generate indices for each trial
            for c = 1:nChunks
                chunkInds=1:(length(samples.y(1,1,1,1,:))/nChunks);
                trialInds = trialInds_all(c*chunkLength-(chunkLength-1):c*chunkLength);

                choice(i,c,chunkInds)=samples.y(1,1,i,1,trialInds);%assign to temporary variables

                dx1(i,c,chunkInds)=samples.dx1(1,1,i,1,trialInds);%assign changes in wealth dx for outcome 1 (note same amount for all trials)
                dx2(i,c,chunkInds)=samples.dx2(1,1,i,1,trialInds);%same for outcome 2 etc.
                dx3(i,c,chunkInds)=samples.dx3(1,1,i,1,trialInds);
                dx4(i,c,chunkInds)=samples.dx4(1,1,i,1,trialInds);

                p_a1(i,c,chunkInds)=samples.pa1(1,1,i,1,trialInds);%assign changes in 'probability' for outcome 1
                p_b1(i,c,chunkInds)=samples.pb1(1,1,i,1,trialInds);
            end 
        end 
end
%% Configure data structure for graphical model & parameters to monitor
%everything you want JAGS to use
switch mode
    case {1} %Simulating CPT choices
        dataStruct = struct(...
            'nGambles',nGambles,'nAgents', nAgents,'nChunks',nChunks,'nTrials',nTrials,...
            'dx1',dx1,'dx2',dx2,'dx3',dx3,'dx4',dx4,...
            'pa1',p_a1,'pb1',p_b1,...
            'muLogBetaL',muLogBetaL,'muLogBetaU',muLogBetaU,'sigmaLogBetaL',sigmaLogBetaL,'sigmaLogBetaU',sigmaLogBetaU,...
            'muLogAlphaL',muLogAlphaL,'muLogAlphaU',muLogAlphaU,'sigmaLogAlphaL',sigmaLogAlphaL,'sigmaLogAlphaU',sigmaLogAlphaU,...
            'muLogDeltaL',muLogDeltaL,'muLogDeltaU',muLogDeltaU,'sigmaLogDeltaL',sigmaLogDeltaL,'sigmaLogDeltaU',sigmaLogDeltaU,...
            'muLogGammaL',muLogGammaL,'muLogGammaU',muLogGammaU,'sigmaLogGammaL',sigmaLogGammaL,'sigmaLogGammaU',sigmaLogGammaU,...
            'pz',pz);
        
    case {2} %Simulating LML choices
        dataStruct = struct(...
            'nGambles',nGambles,'nAgents', nAgents,'nChunks',nChunks,'nTrials',nTrials,...
            'dx1',dx1,'dx2',dx2,'dx3',dx3,'dx4',dx4,...
            'pa1',p_a1,'pb1',p_b1,...
            'muLogBetaL',muLogBetaL,'muLogBetaU',muLogBetaU,'sigmaLogBetaL',sigmaLogBetaL,'sigmaLogBetaU',sigmaLogBetaU,...
            'muLogAlphaL',muLogAlphaL,'muLogAlphaU',muLogAlphaU,'sigmaLogAlphaL',sigmaLogAlphaL,'sigmaLogAlphaU',sigmaLogAlphaU,...
            'muLogDeltaL',muLogDeltaL,'muLogDeltaU',muLogDeltaU,'sigmaLogDeltaL',sigmaLogDeltaL,'sigmaLogDeltaU',sigmaLogDeltaU,...
            'muLogGammaL',muLogGammaL,'muLogGammaU',muLogGammaU,'sigmaLogGammaL',sigmaLogGammaL,'sigmaLogGammaU',sigmaLogGammaU,...
            'pz',pz);
        
    case {3} %Model recovery CPT data
        dataStruct = struct(...
            'nGambles',nGambles,'nAgents', nAgents,'nChunks',nChunks,'nTrials',nTrials,'y',choice,...
            'dx1',dx1,'dx2',dx2,'dx3',dx3,'dx4',dx4,...
            'pa1',p_a1,'pb1',p_b1,...
            'muLogBetaL',muLogBetaL,'muLogBetaU',muLogBetaU,'sigmaLogBetaL',sigmaLogBetaL,'sigmaLogBetaU',sigmaLogBetaU,...
            'muLogAlphaL',muLogAlphaL,'muLogAlphaU',muLogAlphaU,'sigmaLogAlphaL',sigmaLogAlphaL,'sigmaLogAlphaU',sigmaLogAlphaU,...
            'muLogDeltaL',muLogDeltaL,'muLogDeltaU',muLogDeltaU,'sigmaLogDeltaL',sigmaLogDeltaL,'sigmaLogDeltaU',sigmaLogDeltaU,...
            'muLogGammaL',muLogGammaL,'muLogGammaU',muLogGammaU,'sigmaLogGammaL',sigmaLogGammaL,'sigmaLogGammaU',sigmaLogGammaU,...
            'pz',pz);
        
    case {4} %Model recovery LML data
        dataStruct = struct(...
            'nGambles',nGambles,'nAgents', nAgents,'nChunks',nChunks,'nTrials',nTrials,'y',choice,...
            'dx1',dx1,'dx2',dx2,'dx3',dx3,'dx4',dx4,...
            'pa1',p_a1,'pb1',p_b1,...
            'muLogBetaL',muLogBetaL,'muLogBetaU',muLogBetaU,'sigmaLogBetaL',sigmaLogBetaL,'sigmaLogBetaU',sigmaLogBetaU,...
            'muLogAlphaL',muLogAlphaL,'muLogAlphaU',muLogAlphaU,'sigmaLogAlphaL',sigmaLogAlphaL,'sigmaLogAlphaU',sigmaLogAlphaU,...
            'muLogDeltaL',muLogDeltaL,'muLogDeltaU',muLogDeltaU,'sigmaLogDeltaL',sigmaLogDeltaL,'sigmaLogDeltaU',sigmaLogDeltaU,...
            'muLogGammaL',muLogGammaL,'muLogGammaU',muLogGammaU,'sigmaLogGammaL',sigmaLogGammaL,'sigmaLogGammaU',sigmaLogGammaU,...
            'pz',pz);
        
    case {5} %Parameter recovery for CPT data
        dataStruct = struct(...
            'nGambles',nGambles,'nAgents', nAgents,'nChunks',nChunks,'nTrials',chunkLength,'y',choice,...
            'dx1',dx1,'dx2',dx2,'dx3',dx3,'dx4',dx4,...
            'pa1',p_a1,'pb1',p_b1,...
            'muLogBetaL',muLogBetaL,'muLogBetaU',muLogBetaU,'sigmaLogBetaL',sigmaLogBetaL,'sigmaLogBetaU',sigmaLogBetaU,...
            'muLogAlphaL',muLogAlphaL,'muLogAlphaU',muLogAlphaU,'sigmaLogAlphaL',sigmaLogAlphaL,'sigmaLogAlphaU',sigmaLogAlphaU,...
            'muLogDeltaL',muLogDeltaL,'muLogDeltaU',muLogDeltaU,'sigmaLogDeltaL',sigmaLogDeltaL,'sigmaLogDeltaU',sigmaLogDeltaU,...
            'muLogGammaL',muLogGammaL,'muLogGammaU',muLogGammaU,'sigmaLogGammaL',sigmaLogGammaL,'sigmaLogGammaU',sigmaLogGammaU,...
            'pz',pz);
        
    case {6} %Parameter recovery for LML data
        dataStruct = struct(...
            'nGambles',nGambles,'nAgents', nAgents,'nChunks',nChunks,'nTrials',chunkLength,'y',choice,...
            'dx1',dx1,'dx2',dx2,'dx3',dx3,'dx4',dx4,...
            'pa1',p_a1,'pb1',p_b1,...
            'muLogBetaL',muLogBetaL,'muLogBetaU',muLogBetaU,'sigmaLogBetaL',sigmaLogBetaL,'sigmaLogBetaU',sigmaLogBetaU,...
            'muLogAlphaL',muLogAlphaL,'muLogAlphaU',muLogAlphaU,'sigmaLogAlphaL',sigmaLogAlphaL,'sigmaLogAlphaU',sigmaLogAlphaU,...
            'muLogDeltaL',muLogDeltaL,'muLogDeltaU',muLogDeltaU,'sigmaLogDeltaL',sigmaLogDeltaL,'sigmaLogDeltaU',sigmaLogDeltaU,...
            'muLogGammaL',muLogGammaL,'muLogGammaU',muLogGammaU,'sigmaLogGammaL',sigmaLogGammaL,'sigmaLogGammaU',sigmaLogGammaU,...
            'pz',pz);
end

for i = 1:nChains
    switch mode  
        case {1}  %Simulating CPT choices
            monitorParameters = {'dx1','dx2','dx3','dx4','pa1','pb1',...
                'y',...
                'alpha_pt','gamma_pt','delta_pt','beta_pt',...
                'beta_lml', 'alpha_lml'};
            S=struct; init0(i)=S; %sets initial values as empty so randomly seeded
            
        case {2}  %Simulating LML choices
            monitorParameters = {'dx1','dx2','dx3','dx4','pa1','pb1',...
                'y',...
                'alpha_pt','gamma_pt','delta_pt','beta_pt',...
                'beta_lml', 'alpha_lml'};
            S=struct; init0(i)=S; %sets initial values as empty so randomly seeded
            
        case {3}  %Model recovery CPT data
            monitorParameters = {'dx1','dx2','dx3','dx4','pa1','pb1',...
                'y','z',...
                'alpha_pt','gamma_pt','delta_pt','beta_pt',...
                'beta_lml', 'alpha_lml'};
            S=struct; init0(i)=S; %sets initial values as empty so randomly seeded
            
        case {4}  %Model recovery LML data
            monitorParameters = {'dx1','dx2','dx3','dx4','pa1','pb1',...
                'y','z'...
                'alpha_pt','gamma_pt','delta_pt','beta_pt',...
                'beta_lml', 'alpha_lml'};
            S=struct; init0(i)=S; %sets initial values as empty so randomly seeded
            
        case {5}  %Parameter recovery for CPT data
            monitorParameters = {'dx1','dx2','dx3','dx4','pa1','pb1',...
                'y',...
                'alpha_pt','gamma_pt','delta_pt','beta_pt',...
                'beta_lml', 'alpha_lml'};
            S=struct; init0(i)=S; %sets initial values as empty so randomly seeded

        case {6}  %Parameter recovery for LML data
            monitorParameters = {'dx1','dx2','dx3','dx4','pa1','pb1',...
                'y',...
                'alpha_pt','gamma_pt','delta_pt','beta_pt',...
                'beta_lml', 'alpha_lml'};
            S=struct; init0(i)=S; %sets initial values as empty so randomly seeded
                      
    end
end

%% Run JAGS sampling via matJAGS
tic;
fprintf( 'Running JAGS ...\n' ); % start clock to time % display

[samples, stats] = matjags( ...
    dataStruct, ...                           % Struct that contains all relevant data
    fullfile(jagsDir, [modelName '.txt']), ...% File that contains model definition
    init0, ...                                % Initial values for latent variables
    whichJAGS,...                             % Specifies which copy of JAGS to run on
    'doparallel' , doParallel, ...            % Parallelization flag
    'nchains', nChains,...                    % Number of MCMC chains
    'nburnin', nBurnin,...                    % Number of burnin steps
    'nsamples', nSamples, ...                 % Number of samples to extract
    'thin', nThin, ...                        % Thinning parameter
    'dic', doDIC, ...                         % Do the DIC?
    'monitorparams', monitorParameters, ...   % List of latent variables to monitor
    'savejagsoutput' , 1 , ...                % Save command line output produced by JAGS?
    'verbosity' , 1 , ...                     % 0=do not produce any output; 1=minimal text output; 2=maximum text output
    'cleanup' , 0 ,...                        % clean up of temporary files?
    'rndseed',1);                             % Randomise seed; 0=no; 1=yes

toc % end clock

%% Save stats and samples
disp('saving samples and stats...')
%save(['samples_stats\', modelName,'_',priorName,'_',dataSource,'_burn_',num2str(nBurnin),'_samps_',num2str(nSamples),'_chains_',num2str(nChains),'_',datestr(now,'mm-dd-yyyy HH-MM')],'stats','samples','-v7.3')
save(['samples_stats\',sprintf('%s_Gamble_%.0f',outputName,g)],'stats','samples','-v7.3')

%['samples_stats\', outputName," Gamble_",int2str(g)],'stats','samples','-v7.3')
end %and wrapping of gambles

%% Print readouts
disp('stats:'),disp(stats)%print out structure of stats output
disp('samples:'),disp(samples);%print out structure of samples output
%try
%    rhats=fields(stats.Rhat);
%    for lp = 1: length(rhats)
%        disp(['stats.Rhat.',rhats{lp}]);
%        eval(strcat('stats.Rhat.',rhats{lp}))
%    end
%catch
%    disp('no field for stats.Rhat')
end