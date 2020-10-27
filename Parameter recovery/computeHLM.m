function computeHLM(runModelNum,nBurnin,nSamples,nThin,nChains,subjList,whichJAGS,Mode,doParallel)

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
%Set model name, number of utility models, and prior over model indicator
%variable z
switch runModelNum
    case 1 %parameter recovery
        modelName = 'JAGS_models_Subjectwise'; priorName='NoPriorNeeded';
end

%% Set key variables
nTrials=100;%each session have 100 trials
nSessions=10;
doDIC=1;%compute Deviance information criteria? This is the hierarchical equivalent of an AIC, the lower the better
nSubjects=length(subjList);%number of subjects

%% Set bounds of hyperpriors
%hard code the upper and lower bounds of hyperpriors, typically uniformly
%distributed in log space. These values will be imported to JAGS.

%%%%%NOT CORRECT YET!
if mode==1 %fixed values

%beta - prior on log since cannot be less than 0; note same bounds used for independent priors on all utility models
muLogBetaL=-2.3;muLogBetaU=3.4;muLogBetaM=(muLogBetaL+muLogBetaU)/2; %bounds on mean of distribution log beta
sigmaLogBetaL=0.01;sigmaLogBetaU=sqrt(((muLogBetaU-muLogBetaL)^2)/12);sigmaLogBetaM=(sigmaLogBetaL+sigmaLogBetaU)/2;%bounds on the std of distribution of log beta

%Alpha - prior on log since cannot be less than 0
muLogAlphaL=-2.3;muLogAlphaU=0;muLogAlphaM=(muLogAlphaL+muLogAlphaU)/2;%bounds on mean of distribution of log Alpha
sigmaLogAlphaL=0.01;sigmaLogAlphaU=sqrt(((muLogAlphaU-muLogAlphaL)^2)/12);sigmaLogAlphaM=(sigmaLogAlphaL+sigmaLogAlphaU)/2; %bounds on std of distribution of log Alpha

%Lambda - prior on log since cannot be less than 0
muLogLambdaL=0;muLogLambdaU=1.6;muLogLambdaM=(muLogLambdaL+muLogLambdaU)/2;%bounds on the mean of distribution of log Lambda
sigmaLogLambdaL=0.01;sigmaLogLambdaU=sqrt(((muLogLambdaU-muLogLambdaL)^2)/12);sigmaLogLambdaM=(sigmaLogLambdaL+sigmaLogLambdaU)/2;%bounds on the std of log Lambda

%eta
muEtaL=-2.5;muEtaU=2.5;muEtaM=(muEtaL+muEtaU)/2;%bounds on mean of distribution of eta
sigmaEtaL=0.01;sigmaEtaU=sqrt(((muEtaU-muEtaL)^2)/12);sigmaEtaM=(sigmaEtaL+sigmaEtaU)/2;%bounds on std of eta

%weight
weight_beta_L=3; weight_beta_U=4;
weight_alpha_L=2; weight_alpha_U=3;


%% Print information for user
disp('**************');
disp(['running model#_',num2str(runModelNum),':'])
disp([modelName,'_started:_',datestr(clock)])
disp(['MCMC number_',num2str(whichJAGS)])
disp(['running on_',dataSource])
disp(['with_',priorName])
disp('**************');

%% Initialise matrices

%%%%%NOT DONE

%initialise matrices with nan values of size subjects x conditions x trials
choice = nan(nSubjects,nConditions,maxTrials); %initialise choice data matrix 
dx1 = nan(nSubjects,nConditions,maxTrials); dx2 = dx1; dx3 = dx1; dx4=dx1;%initialise changes in wealth
out1= dx1;out2=dx1;out3=dx1;out4=dx1;%initialise outcomes
deuLin= nan(nSubjects,nConditions,maxTrials);deuLog= nan(nSubjects,nConditions,maxTrials);%initialise changes in linear utility / log utility

%% Compile choice & gamble data


%%%DO I NEED THIS?

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

%NOT DONE (NOTHING BELOW NEITHER)

wealths=[Wealth_add;Wealth_multi]
wealths=wealths(:,subjList) %limits to subjects in subjlist

%% Truncate data
%truncate down to 312 trials
truncInds=1:nTrials;
choice=choice(:,:,truncInds);
dx1=dx1(:,:,truncInds);dx2=dx2(:,:,truncInds);dx3=dx3(:,:,truncInds);dx4=dx4(:,:,truncInds);
deuLin=deuLin(:,:,truncInds);deuLog=deuLog(:,:,truncInds);

%% Add gamble data for missing trials
%sub 2 multi had only 299 trials, therefore we add random gambles to pad out to 312. this
%allows jags to work since doesn't work for partial observation. this does not affect
% parameter estimation. nans in the choice data are allowed as long as all covariates are not nan.
if nConditions==2 && any(subjList==2)
    dx1(2,2,300:312)=dx1(2,2,1:13);dx2(2,2,300:312)=dx2(2,2,1:13);
    dx3(2,2,300:312)=dx3(2,2,1:13);dx4(2,2,300:312)=dx4(2,2,1:13);
    deuLin(2,2,300:312)=deuLin(2,2,1:13);
    deuLog(2,2,300:312)=deuLog(2,2,1:13);
end

%% Nan check
disp([num2str(length(find(isnan(choice)))),'_nans in choice data']);%nans in choice data do not matter
disp([num2str(length(find(isnan(dx1)))),'_nans in gambles 1 matrix'])% nans in gamble matrices do, since model will not run
disp([num2str(length(find(isnan(dx2)))),'_nans in gambles 2 matrix'])
disp([num2str(length(find(isnan(dx3)))),'_nans in gambles 3 matrix'])
disp([num2str(length(find(isnan(dx4)))),'_nans in gambles 4 matrix'])
disp([num2str(length(find(isnan(deuLin)))),'_nans in deu_lin'])
disp([num2str(length(find(isnan(deuLog)))),'_nans in deu_log'])

%% Visualise changes in wealth
if runPlots
    figure;subplot(2,1,1),histogram([(dx1(:,1,:)+dx2(:,1,:))/2,(dx3(:,1,:)+dx4(:,1,:))/2],'Normalization','pdf')
    hold on,histogram([(dx1(:,2,:)+dx2(:,2,:))/2,(dx3(:,2,:)+dx4(:,2,:))/2],'Normalization','pdf');
    ntitle('EV of dx - blue: add, red: mult')
    subplot(2,1,2)
    tmp1=(out1(:,1,:)+out2(:,1,:))/2;tmp2=(out3(:,1,:)+out4(:,1,:))/2;data1=[tmp1(:);tmp2(:)];
    tmp1=(out1(:,2,:)+out2(:,2,:))/2;tmp2=(out3(:,2,:)+out4(:,2,:))/2;data2=[tmp1(:);tmp2(:)];
    nBins1=15;nBins2=15;
    hAx1 = gca;% create one axis or get current axes
    posAx1 = get(hAx1, 'Position');% get position of axis
    hAx2 = axes('Position', posAx1);% create an overlapping axis at the same location
    h1=histogram(hAx1,data1,nBins1,'Normalization','probability');% histogram for first data vector
    h2=histogram(hAx2,data2,nBins2,'Normalization','probability','FaceColor','r');
    set(hAx2,'Color','none');% make second axis transparent
    ylabel(hAx1,'Probability');% ylabel for histogram 1
    set(hAx2,'XAxisLocation','top');
    xlabel(hAx2,'Time average multiplicative growth');xlabel(hAx1,'Time average additive growth');set(hAx1,'XColor','r');set(hAx2,'XColor','b')
end

%% Compute Time averages for eta beta space
if runPlots
    nEtas=100;nBetas=100;TimeAvAdd=[];TimeAvMult=[];%number of etas and betas to simulate; initialise
    etaParam=linspace(muEtaL,muEtaU,nEtas);betaParam=linspace(exp(muLogBetaL),exp(muLogBetaU),nBetas);
    for e=1:nEtas
        for b=1:nBetas
            [TimeAvAdd(e,b),TimeAvMult(e,b)]=computeEtaBeta2TimeAv(etaParam(e),betaParam(b),wealths,dx1,dx2,dx3,dx4);
        end
    end
    [etaVals,betaVals]=meshgrid(betaParam,etaParam);
    figure,subplot(2,2,1),surf(etaVals,betaVals,TimeAvMult),title('mult');ylabel('beta'),xlabel('eta'),zlabel('Time av');
    subplot(2,2,2),surf(etaVals,betaVals,TimeAvAdd),title('add');ylabel('beta'),xlabel('eta'),zlabel('Time av');
    subplot(2,2,3),yyaxis right,plot(etaParam,TimeAvMult(:,1),'r'),ylabel('Time Average Multiplicative Growth Rate')
    yyaxis left,plot(etaParam,TimeAvAdd(:,1),'b'),xlabel('eta'),ylabel('Time Average Additive Growth Rate')
    line([0, 0], ylim, 'LineWidth', 1.5, 'Color', 'b','LineStyle','--'),line([1, 1], ylim, 'LineWidth', 1.5, 'Color', 'r','LineStyle','--')
    subplot(2,2,4),yyaxis right,plot(etaParam,(TimeAvMult(:,nBetas)),'r'),ylabel('Time Average Multiplicative Growth Rate')
    yyaxis left,plot(etaParam,(TimeAvAdd(:,nBetas)),'b'),xlabel('eta'),ylabel('Time Average Additive Growth Rate')
    line([0, 0], ylim, 'LineWidth', 1.5, 'Color', 'b','LineStyle','--'),line([1, 1], ylim, 'LineWidth', 1.5, 'Color', 'r','LineStyle','--')
    save('data_etaBeta2Time','TimeAvAdd','TimeAvMult','-v7.3') 
end

%% Configure data structure for graphical model & parameters to monitor
%everything you want jags to use
switch runModelNum
    case {1} %Parameter estimation of eta
        dataStruct = struct(...
            'wealths',wealths,'nSubjects', nSubjects,'nConditions',nConditions,...
            'nTrials',nTrials,'dx1',dx1,'dx2',dx2,'dx3',dx3,'dx4',dx4,'y',choice,...
            'muLogBetaL',muLogBetaL,'muLogBetaU',muLogBetaU,'sigmaLogBetaL',sigmaLogBetaL,'sigmaLogBetaU',sigmaLogBetaU,...
            'muEtaL',muEtaL,'muEtaU',muEtaU,'sigmaEtaL',sigmaEtaL,'sigmaEtaU',sigmaEtaU);
        
    case {2}
        dataStruct = struct(...
            'wealths',wealths,'nSubjects', nSubjects,'nConditions',nConditions,...
            'nTrials',nTrials,'dx1',dx1,'dx2',dx2,'dx3',dx3,'dx4',dx4,...
            'y',choice,'muLogBetaL',muLogBetaL,...
            'muLogBetaU',muLogBetaU,'sigmaLogBetaL',sigmaLogBetaL,'sigmaLogBetaU',sigmaLogBetaU,...
            'muLogAlphaL',muLogAlphaL,'muLogAlphaU',muLogAlphaU,'sigmaLogAlphaL',sigmaLogAlphaL,...
            'sigmaLogAlphaU',sigmaLogAlphaU,'muLogLambdaL',muLogLambdaL,'muLogLambdaU',muLogLambdaU,...
            'sigmaLogLambdaL',sigmaLogLambdaL,'sigmaLogLambdaU',sigmaLogLambdaU,'muEtaL',muEtaL,...
            'muEtaU',muEtaU,'sigmaEtaL',sigmaEtaL,'sigmaEtaU',sigmaEtaU,...
            'pz',pz);
        
    case {3}
        dataStruct = struct(...
            'wealths',wealths,'nSubjects', nSubjects,'nConditions',nConditions,...
            'nTrials',nTrials,'dx1',dx1,'dx2',dx2,'dx3',dx3,'dx4',dx4,...
            'y',choice,'muLogBetaL',muLogBetaL,...
            'muLogBetaU',muLogBetaU,'sigmaLogBetaL',sigmaLogBetaL,'sigmaLogBetaU',sigmaLogBetaU,...
            'muLogAlphaL',muLogAlphaL,'muLogAlphaU',muLogAlphaU,'sigmaLogAlphaL',sigmaLogAlphaL,...
            'sigmaLogAlphaU',sigmaLogAlphaU,'muLogLambdaL',muLogLambdaL,'muLogLambdaU',muLogLambdaU,...
            'sigmaLogLambdaL',sigmaLogLambdaL,'sigmaLogLambdaU',sigmaLogLambdaU,...
            'weight_beta_L', weight_beta_L, 'weight_beta_U', weight_beta_U,...
            'weight_alpha_L', weight_alpha_L, 'weight_alpha_U', weight_alpha_U, 'pz',pz);
    
    case {4}
        dataStruct = struct(...
            'wealths',wealths,'nSubjects', nSubjects,'nConditions',nConditions,...
            'nTrials',nTrials,'dx1',dx1,'dx2',dx2,'dx3',dx3,'dx4',dx4,...
            'y',choice,'muLogBetaL',muLogBetaL,...
            'muLogBetaU',muLogBetaU,'sigmaLogBetaL',sigmaLogBetaL,'sigmaLogBetaU',sigmaLogBetaU,...
            'muLogAlphaL',muLogAlphaL,'muLogAlphaU',muLogAlphaU,'sigmaLogAlphaL',sigmaLogAlphaL,...
            'sigmaLogAlphaU',sigmaLogAlphaU,'muLogLambdaL',muLogLambdaL,'muLogLambdaU',muLogLambdaU,...
            'sigmaLogLambdaL',sigmaLogLambdaL,'sigmaLogLambdaU',sigmaLogLambdaU,...
            'weight_beta_L', weight_beta_L, 'weight_beta_U', weight_beta_U,...
            'weight_alpha_L', weight_alpha_L, 'weight_alpha_U', weight_alpha_U, 'pz',pz);
        
    case {5} 
        dataStruct = struct(...
            'wealths',wealths,'nSubjects', nSubjects,'nConditions',nConditions,...
            'nTrials',nTrials,'dx1',dx1,'dx2',dx2,'dx3',dx3,'dx4',dx4,...
            'y',choice,'muLogBetaL',muLogBetaL,...
            'muLogBetaU',muLogBetaU,'sigmaLogBetaL',sigmaLogBetaL,'sigmaLogBetaU',sigmaLogBetaU,...
            'muLogAlphaL',muLogAlphaL,'muLogAlphaU',muLogAlphaU,'sigmaLogAlphaL',sigmaLogAlphaL,...
            'sigmaLogAlphaU',sigmaLogAlphaU,'muLogLambdaL',muLogLambdaL,'muLogLambdaU',muLogLambdaU,...
            'sigmaLogLambdaL',sigmaLogLambdaL,'sigmaLogLambdaU',sigmaLogLambdaU,'muEtaL',muEtaL,...
            'muEtaU',muEtaU,'sigmaEtaL',sigmaEtaL,'sigmaEtaU',sigmaEtaU,...
            'pz',pz);
        
    case {6} 
        dataStruct = struct(...
            'wealths',wealths,'nSubjects', nSubjects,'nConditions',nConditions,...
            'nTrials',nTrials,'dx1',dx1,'dx2',dx2,'dx3',dx3,'dx4',dx4,...
            'y',choice,'muLogBetaL',muLogBetaL,...
            'muLogBetaU',muLogBetaU,'sigmaLogBetaL',sigmaLogBetaL,'sigmaLogBetaU',sigmaLogBetaU,...
            'muLogAlphaL',muLogAlphaL,'muLogAlphaU',muLogAlphaU,'sigmaLogAlphaL',sigmaLogAlphaL,...
            'sigmaLogAlphaU',sigmaLogAlphaU,'muLogLambdaL',muLogLambdaL,'muLogLambdaU',muLogLambdaU,...
            'sigmaLogLambdaL',sigmaLogLambdaL,'sigmaLogLambdaU',sigmaLogLambdaU,'muEtaL',muEtaL,...
            'muEtaU',muEtaU,'sigmaEtaL',sigmaEtaL,'sigmaEtaU',sigmaEtaU,...
            'pz',pz);
end

for i = 1:nChains
    switch runModelNum
        
        case {1}  %Parameter estimation
            monitorParameters = {'dx1','dx2','dx3','dx4','beta_iso','eta','mu_eta','sigma_eta','mu_log_beta_iso','sigma_log_beta_iso'};
            S=struct; init0(i)=S; %sets initial values as empty so randomly seeded
            
        case {2} %Model selection for subjectwise strong models
            monitorParameters = {...
                'eta_iso','eta_tw','alphaGain','alphaLoss','lambda',...%utility params
                'beta_tw','beta_pt','beta_iso',...%betas
                'z','px_z1','px_z2','delta_z1','sum_z'};%model indicator
            S=struct; init0(i)=S;   %sets initial values as empty so randomly seeded
                 
        case {3} %Model selection for Prospect theory (original and weighted gain)
            monitorParameters = {...
                'mu_log_beta_p','tau_log_beta_p','sigma_log_beta_p',...
                'mu_log_beta_pg','tau_log_beta_pg','sigma_log_beta_pg',... #beta hyperpriors
                'mu_log_alphaGain_p','tau_log_alphaGain_p','sigma_log_alphaGain_p ',... 
                'mu_log_alphaGain_pg','tau_log_alphaGain_pg','sigma_log_alphaGain_pg ',... #alpha_gain hyperpriors
                'mu_log_alphaLoss_p','tau_log_alphaLoss_p','sigma_log_alphaLoss_p',... #alpha_loss hyperpriors
                'mu_log_lambda_p','tau_log_lambda_p','sigma_log_lambda_p',... #lambda hyperpriors
                'weight_a_pg','weight_b_pg',... #weight hyperpriors
                'alphaGain_p','alphaLoss_p','lambda_p','alphaGain_pg',...%utility params
                'beta_p' ,'beta_pg',...%betas
                'w_pg',...%weights
                'z','px_z1','px_z2','delta_z1','sum_z'};%model indicator
            S=struct; init0(i)=S;   %sets initial values as empty so randomly seeded
            
        case {4} %Parameter retrival for Prospect theory original
            monitorParameters = {...
                'mu_log_beta_p','tau_log_beta_p','sigma_log_beta_p',...
                'mu_log_beta_pg','tau_log_beta_pg','sigma_log_beta_pg',... #beta hyperpriors
                'mu_log_alphaGain_p','tau_log_alphaGain_p','sigma_log_alphaGain_p ',... 
                'mu_log_alphaGain_pg','tau_log_alphaGain_pg','sigma_log_alphaGain_pg ',... #alpha_gain hyperpriors
                'mu_log_alphaLoss_p','tau_log_alphaLoss_p','sigma_log_alphaLoss_p',... #alpha_loss hyperpriors
                'mu_log_lambda_p','tau_log_lambda_p','sigma_log_lambda_p',... #lambda hyperpriors
                'weight_a_pg','weight_b_pg',... #weight hyperpriors
                'alphaGain_p','alphaLoss_p','lambda_p','alphaGain_pg',...%utility params
                'beta_p' ,'beta_pg',...%betas
                'w_pg',...%weights
                'z','px_z1','px_z2','delta_z1','sum_z'};%model indicator
            S=struct; init0(i)=S;   %sets initial values as empty so randomly seeded

            
        case {5} %Model selection for subjectwise strong models (Including Prospect theory weighted gain)
            monitorParameters = {...
                'mu_log_beta_tw','tau_log_beta_tw','sigma_log_beta_tw',...
                'mu_log_beta_pt','tau_log_beta_pt','sigma_log_beta_pt',...
                'mu_log_beta_pg','tau_log_beta_pg','sigma_log_beta_pg',... 
                'mu_log_beta_iso','tau_log_beta_iso','sigma_log_beta_iso',... #beta hyperpriors
                'mu_eta_tw','tau_eta_tw','sigma_eta_tw',... 
                'mu_eta_iso','tau_eta_iso','sigma_eta_iso',...#eta hyperpriors
                'mu_log_alphaGain','tau_log_alphaGain','sigma_log_alphaGain',... 
                'mu_log_alphaGain_pg','tau_log_alphaGain_pg','sigma_log_alphaGain_pg ',... #alpha_gain hyperpriors
                'mu_log_alphaLoss','tau_log_alphaLoss','sigma_log_alphaLoss',... #alpha_loss hyperpriors
                'mu_log_lambda','tau_log_lambda','sigma_log_lambda',... #lambda hyperpriors
                'weight_a_pg','weight_b_pg',... #weight hyperpriors
                'eta_iso','eta_tw','alphaGain','alphaLoss','lambda','alphaGain_pg',...%utility params
                'beta_tw','beta_pt','beta_pg','beta_iso',...%betas
                'w_pg',...
                'z','px_z1','px_z2','delta_z1','sum_z'};%model indicator
            S=struct; init0(i)=S;   %sets initial values as empty so randomly seeded
            
        case {6} %Model selection for subjectwise strong models (Including Prospect theory weighted gain)
            monitorParameters = {...
                'mu_log_beta_tw','tau_log_beta_tw','sigma_log_beta_tw',...
                'mu_log_beta_pt','tau_log_beta_pt','sigma_log_beta_pt',...
                'mu_log_beta_pg','tau_log_beta_pg','sigma_log_beta_pg',... 
                'mu_log_beta_iso','tau_log_beta_iso','sigma_log_beta_iso',... #beta hyperpriors
                'mu_eta_tw','tau_eta_tw','sigma_eta_tw',... 
                'mu_eta_iso','tau_eta_iso','sigma_eta_iso',...#eta hyperpriors
                'mu_log_alphaGain','tau_log_alphaGain','sigma_log_alphaGain',... 
                'mu_log_alphaGain_pg','tau_log_alphaGain_pg','sigma_log_alphaGain_pg ',... #alpha_gain hyperpriors
                'mu_log_alphaLoss','tau_log_alphaLoss','sigma_log_alphaLoss',... #alpha_loss hyperpriors
                'mu_log_lambda','tau_log_lambda','sigma_log_lambda',... #lambda hyperpriors
                'weight_a_pg','weight_b_pg',... #weight hyperpriors
                'eta_iso','eta_tw','alphaGain','alphaLoss','lambda','alphaGain_pg',...%utility params
                'beta_tw','beta_pt','beta_pg','beta_iso',...%betas
                'w_pg',...
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