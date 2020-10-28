function computeHLM(runModelNum,nBurnin,nSamples,nThin,nChains,subjList,whichJAGS,Mode,doParallel,whichGamble)

%% Hiercharchical Latent Mixture (HLM) model
% This is a general script for running several types of hierarchical
% bayesian model via JAGS. It can run hiearchical latent mixture models in
% which different utility models can be compared via inference on the model
% indicator variables, and it can run without latent mixtures of models for
% instance in order to estimate parameters of a given utility model. It
% takes as input the following:

% runModelNum - a number which selects which JAGS model to run.
% nBurnin - a number which specifies how many burn in samples to run.
% nThin - a number specifying the thinnning number
% nChains - number of chains
% subjList - list of subject numbers to include
% whichJAGS - sets which copy of matjags to run
% synthMode - sets whether to run on real data (1), synthetic data for model recovery (2) or parameter recovery (3)

%% Set paths
[startDir,~] = fileparts(mfilename('fullpath'));%specify your starting directory here (where this script runs from)
cd(startDir);%move to starting directory
jagsDir=[startDir,'/JAGS'];
addpath(fullfile(pwd,'/parameter_recovert/matjags'));%set path to include matjags folder
addpath(fullfile(pwd,'/parameter_recovery/data'));%set path to include data folder

%% Choose & load data

%%%%%% IF MODE 1 THEN NO DATA
switch Mode
    case 1, dataSource ='allData'; %Real experimental data
    case 2, dataSource ='allData_synth_modelRecov';%Synthetic data for model recovery from 2 different models
end
load(dataSource)

%% Set model specific variables
%Set model name (if running model comparisson then also set prior over
%model indicator variable z)

switch runModelNum
    case 1 %simulating choices
        modelName = 'JAGS_models_Subjectwise_simulation'; priorName='';
        
    case 2 %parameter recovery
        modelName = 'JAGS_models_Subjectwise_recovery'; priorName='';
end

%% Set key variables
nTrials=100;%All trials (nTrials*nSamples)
doDIC=1;%compute Deviance information criteria? This is the hierarchical equivalent of an AIC, the lower the better
nSubjects=length(subjList);%number of subjects

%% Set bounds of hyperpriors
%hard code the upper and lower bounds of hyperpriors, typically uniformly
%distributed in log space. These values will be imported to JAGS.

%%%%%%%%%%%% NOTE FOR GAMBLE SIMULATIONS THESE ARE HARDCODED INTO JAGS! %%%%%%%%%%%%

%beta - prior on log since cannot be less than 0; note same bounds used for independent priors on all utility models
muLogBetaL=-2.3;muLogBetaU=3.4;muLogBetaM=(muLogBetaL+muLogBetaU)/2; %bounds on mean of distribution log beta
sigmaLogBetaL=0.01;sigmaLogBetaU=sqrt(((muLogBetaU-muLogBetaL)^2)/12);sigmaLogBetaM=(sigmaLogBetaL+sigmaLogBetaU)/2;%bounds on the std of distribution of log beta

%Alpha - prior on log since cannot be less than 0; note same bounds used for independent priors on all utility models
muLogAlphaL=-2.3;muLogAlphaU=0;muLogAlphaM=(muLogAlphaL+muLogAlphaU)/2;%bounds on mean of distribution of log Alpha
sigmaLogAlphaL=0.01;sigmaLogAlphaU=sqrt(((muLogAlphaU-muLogAlphaL)^2)/12);sigmaLogAlphaM=(sigmaLogAlphaL+sigmaLogAlphaU)/2; %bounds on std of distribution of log Alpha

%Gamma

%Delta


%% Print information for user
disp('**************');
disp(['running model#_',num2str(runModelNum),':'])
disp([modelName,'_started:_',datestr(clock)])
disp(['MCMC number_',num2str(whichJAGS)])
disp(['running on_',dataSource])
disp(['with_',priorName])
disp('**************');

%% Initialise matrices
%initialise matrices with nan values of size subjects x conditions x trials
choice = nan(nSubjects,nTrials); %initialise choice data matrix 
dx1 = nan(nSubjects,nTrials); dx2 = dx1; dx3 = dx1; dx4=dx1;%initialise changes in wealth
p1 = nan(nSubjects,nTrials); p2 = p1; p3 = p1; p4 = p1; %initialuse channges in 'probabilities'
out1= dx1;out2=dx1;out3=dx1;out4=dx1;%initialise outcomes
deuLin= nan(nSubjects,maxTrials);deuLog= nan(nSubjects,nConditions,maxTrials);%initialise changes in linear utility / log utility

%% Compile choice & gamble data


%%IT IS HERE THE DATA INPUT NEEDS TO BE!

% Jags cannot deal with partial observations, so we need to specify gamble info for all nodes. This doesn't change anything.
for i = 1:nSubjects
    for c = 1:nConditions
        switch c %condition
            case 1% add
                trialInds=1:length(Choice_add{subjList(i)});%generate indices for each trial
                choice(i,c,trialInds)=Choice_add{subjList(i)}(trialInds);%assign to temporary variables
                dx1(i,c,trialInds)=LinU_Gam1_1_add{subjList(i)}(trialInds);%assign changes in wealth dx for outcome 1
                dx2(i,c,trialInds)=LinU_Gam1_2_add{subjList(i)}(trialInds);%same for outcome 2 etc.
                dx3(i,c,trialInds)=LinU_Gam2_1_add{subjList(i)}(trialInds);
                dx4(i,c,trialInds)=LinU_Gam2_2_add{subjList(i)}(trialInds);
                out1(i,c,trialInds)=dx1(i,c,trialInds);out2(i,c,trialInds)=dx2(i,c,trialInds);%specify as outcomes 1 to 4
                out3(i,c,trialInds)=dx3(i,c,trialInds);out4(i,c,trialInds)=dx4(i,c,trialInds);
                deuLin(i,c,trialInds)=delta_EU_Lin_add{subjList(i)}(trialInds);%specify changes in expected utility for each gamble for linear utility
                deuLog(i,c,trialInds)=delta_EU_Log_add{subjList(i)}(trialInds);%specify changes in expected utility for each gamble for log utility
            case 2% multi
                trialInds=1:length(Choice_multi{subjList(i)});
                choice(i,c,trialInds)=Choice_multi{subjList(i)}(trialInds);
                dx1(i,c,trialInds)=LinU_Gam1_1_multi{subjList(i)}(trialInds);
                dx2(i,c,trialInds)=LinU_Gam1_2_multi{subjList(i)}(trialInds);
                dx3(i,c,trialInds)=LinU_Gam2_1_multi{subjList(i)}(trialInds);
                dx4(i,c,trialInds)=LinU_Gam2_2_multi{subjList(i)}(trialInds);
                out1(i,c,trialInds)=LogU_Gam1_1_multi{subjList(i)}(trialInds);out2(i,c,trialInds)=LogU_Gam1_2_multi{subjList(i)}(trialInds);
                out3(i,c,trialInds)=LogU_Gam2_1_multi{subjList(i)}(trialInds);out4(i,c,trialInds)=LogU_Gam2_2_multi{subjList(i)}(trialInds);
                deuLin(i,c,trialInds)=delta_EU_Lin_multi{subjList(i)}(trialInds);
                deuLog(i,c,trialInds)=delta_EU_Log_multi{subjList(i)}(trialInds);
        end
    end
end


%% Compile wealth data

%DON'T UNDERSTAND THIS...

wealths=[Wealth_add];
wealths=wealths(:,subjList); %limits to subjects in subjlist

%% Nan check
disp([num2str(length(find(isnan(choice)))),'_nans in choice data']);%nans in choice data do not matter
disp([num2str(length(find(isnan(dx1)))),'_nans in gambles 1 matrix'])% nans in gamble matrices do, since model will not run
disp([num2str(length(find(isnan(dx2)))),'_nans in gambles 2 matrix'])
disp([num2str(length(find(isnan(dx3)))),'_nans in gambles 3 matrix'])
disp([num2str(length(find(isnan(dx4)))),'_nans in gambles 4 matrix'])
disp([num2str(length(find(isnan(deuLin)))),'_nans in deu_lin'])


%% Configure data structure for graphical model & parameters to monitor
%everything you want jags to use
switch runModelNum
    case {1} %Simulating choices
        dataStruct = struct(...
            'wealths',wealths,'nSubjects', nSubjects,'nTrials',nTrials,...
            'dx1',dx1,'dx2',dx2,'dx3',dx3,'dx4',dx4,...
            'p1',p1,'p2',p2,'p3',p3,'p4',p4);
        
    case {2} %IKKE TAGET STILLING TIL ENDNU 
        dataStruct = struct();
        

end

for i = 1:nChains
    switch runModelNum
        
        case {1}  %Simulating choices
            monitorParameters = {'dx1','dx2','dx3','dx4','p1','p2','p3','p4',...
                'y_pt','alpha_pt','gamma_pt','delta_pt','beta_pt'...
                'y_lml','alpha_lml','gamma_lml','delta_lml','beta_lml'};
            S=struct; init0(i)=S; %sets initial values as empty so randomly seeded
            
        case {2} %IKKE TAGET STILLING TIL ENDNU 
            monitorParameters = {...
                'eta_iso','eta_tw','alphaGain','alphaLoss','lambda',...%utility params
                'beta_tw','beta_pt','beta_iso',...%betas
                'z','px_z1','px_z2','delta_z1','sum_z'};%model indicator
            S=struct; init0(i)=S;   %sets initial values as empty so randomly seeded
               
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
try
    rhats=fields(stats.Rhat);
    for lp = 1: length(rhats)
        disp(['stats.Rhat.',rhats{lp}]);
        eval(strcat('stats.Rhat.',rhats{lp}))
    end
catch
    disp('no field for stats.Rhat')
end