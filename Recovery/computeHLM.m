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

%% Choose & load data

switch mode
    case 1 %simulate choices with CPT
        dataSource = 'all_gambles';
        outputName = 'Choices_simulated_from_CPT'; priorName='';
        pz=[1,0,1,0,1,0,1,0]
        
    case 2 %simulate choices with LML
        dataSource = 'all_gambles';
        outputName = 'Choices_simulated_from_CPT'; priorName='';
        pz=[0,1,0,1,0,1,0,1]
        
    case 3 %Model comparison
        dataSource = %%% MISSING
        outputName = 'model_comparison'; priorName='flat prior';
        pz=repmat(1/8,1,8);
        
    case 4 %parameter recovery for CPT data
        dataSource = %%% MISSING
        outputName = 'parameter_recovery_CPT'; priorName='';
        pz=[1,0,1,0,1,0,1,0]
        
    case 5 %parameter recovery for LML data
        dataSource = %%% MISSING
        outputName = 'parameter_recovery_LML'; priorName='';
        pz=[1,0,1,0,1,0,1,0]

load(dataSource)

%% Set model name
modelName = 'JAGS' %Note same model used for all modes

%% Set key variables
nTrials=length(Data{1,1}.Choice);
doDIC=0;%compute Deviance information criteria? This is the hierarchical equivalent of an AIC, the lower the better

%% Set bounds of hyperpriors

switch mode
    case 1
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


%% Print information for user
disp('**************');
disp(['running model#_',outputName,':'])
disp(['_started:_',datestr(clock)])
disp(['MCMC number_',num2str(whichJAGS)])
disp(['running on_',dataSource])
disp(['with_',priorName])
disp('**************');

%% Initialise matrices
%initialise matrices with nan values of size subjects x conditions x trials
choice = nan(nGambles,nAgents,nTrials); %initialise choice data matrix 
dx1 = nan(nGambles,nAgents,nTrials); dx2 = dx1; dx3 = dx1; dx4=dx1;%initialise changes in wealth
p_a1  = nan(nGambles,nAgents,nTrials); p_a2 = p_a1; p_b1 = p_a1; p_b2 = p_a1; %initialise channges in 'probabilities'

%% Compile choice & gamble data
for g = 1:nGambles
    for i = 1:nAgents
        trialInds=1:length(Data{1,g}.Choice);%generate indices for each trial

        choice(g,i,trialInds)=Data{i,g}.Choice(trialInds);%assign to temporary variables

        dx1(g,i,trialInds)=Data{i,g}.maxA(trialInds);%assign changes in wealth dx for outcome 1 (note same amount for all trials)
        dx2(g,i,trialInds)=Data{i,g}.minA(trialInds);%same for outcome 2 etc.
        dx3(g,i,trialInds)=Data{i,g}.maxB(trialInds);
        dx4(g,i,trialInds)=Data{i,g}.minB(trialInds);

        p_a1(g,i,trialInds)=Data{i,g}.p_maxA(trialInds);%assign changes in 'probability' for outcome 1
        p_b1(g,i,trialInds)=Data{i,g}.p_maxB(trialInds);
    end
end 

%% Configure data structure for graphical model & parameters to monitor
%everything you want jags to use
switch mode
    case {1} %Simulating CPT choices
        dataStruct = struct(...
            'nGambles',nGambles,'nAgents', nAgents,'nTrials',nTrials,...
            'dx1',dx1,'dx2',dx2,'dx3',dx3,'dx4',dx4,...
            'pa1',p_a1,'pb1',p_b1,...
            'muLogBetaU',muLogBetaU,'sigmaLogBetaL',sigmaLogBetaL,'sigmaLogBetaU',sigmaLogBetaU,...
            'muLogAlphaL',muLogAlphaL,'muLogAlphaU',muLogAlphaU,'sigmaLogAlphaL',sigmaLogAlphaL,...
            'muLogDeltaU',muLogDeltaU,'sigmaLogDeltaL',sigmaLogDeltaL,'sigmaLogDeltaU',sigmaLogDeltaU,...
            'muLogGammaL',muLogGammaL,'muLogGammaU',muLogGammaU,'sigmaLogGammaL',sigmaLogGammaL...
            'pz',pz);
        
    case {2} %Simulating LML choices
        dataStruct = struct(...
            'nGambles',nGambles,'nAgents', nAgents,'nTrials',nTrials,...
            'dx1',dx1,'dx2',dx2,'dx3',dx3,'dx4',dx4,...
            'pa1',p_a1,'pb1',p_b1,...
            'muLogBetaU',muLogBetaU,'sigmaLogBetaL',sigmaLogBetaL,'sigmaLogBetaU',sigmaLogBetaU,...
            'muLogAlphaL',muLogAlphaL,'muLogAlphaU',muLogAlphaU,'sigmaLogAlphaL',sigmaLogAlphaL,...
            'muLogDeltaU',muLogDeltaU,'sigmaLogDeltaL',sigmaLogDeltaL,'sigmaLogDeltaU',sigmaLogDeltaU,...
            'muLogGammaL',muLogGammaL,'muLogGammaU',muLogGammaU,'sigmaLogGammaL',sigmaLogGammaL...
            'pz',pz);
        
    case {3} %Model recovery: IKKE TAGET STILLING TIL ENDNU 
        

end

for i = 1:nChains
    switch runModelNum
        
        case {1}  %Simulating CPT choices
            monitorParameters = {'dx1','dx2','dx3','dx4','pa1','pb1',...
                'y','alpha_pt','gamma_pt','delta_pt','beta_pt'};
            S=struct; init0(i)=S; %sets initial values as empty so randomly seeded
            
        case {2} %Simulating CPT choices
            monitorParameters = {'dx1','dx2','dx3','dx4','pa1','pb1','y'};
            S=struct; init0(i)=S; %sets initial values as empty so randomly seeded
            
        case {3} %Model recovery - IKKE TAGET STILLING TIL ENDNU 
               
    end
end

%% Run JAGS sampling via matJAGS
tic;fprintf( 'Running JAGS ...\n' ); % start clock to time % display

[samples, stats] = matjags( ...
    dataStruct, ...                           % Observed data
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
    'cleanup' , 1 ,...                        % clean up of temporary files?
    'rndseed',1);                             % Randomise seed; 0=no; 1=yes

toc % end clock

%% Save stats and samples
disp('saving samples and stats...')
save(['samples_stats\', modelName,'_',priorName,'_',dataSource,'_burn_',num2str(nBurnin),'_samps_',num2str(nSamples),'_chains_',num2str(nChains),'_',datestr(now,'mm-dd-yyyy HH-MM')],'stats','samples','-v7.3')

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