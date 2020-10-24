%% Model Recovery script
%This scripts "presents" synthetic agents with the gambles presented to the
%subjects. The agents operate with utility functions with different
%parameter values. The script generates choice probabilities based on these
%functions, and then probabilistically realises left-right choices. Can
%then be used to feed into HBM script to see whether parameters can be
%recovered

%The script computeSyntheticData4ModelRecovery.m is used to create 
%synthetic agents with different utility functions.
%The script computeSyntheticData4ParameterRecovery.m is used to create 
%synthetic agents with only one type of utility function (isoelastic), but
%where we loop through different combinations of eta parameter values for 
%the two different sessions.

%Script now generates 2 different types of synthetic agents (100 of each), making decisions with different
%utility models (and different parameter values). 
%The file allData_synth_modelRecov.mat will contain choices in following
%order (we use beta = exp(-1)):
% 1) CPT: alpha 0.6, delta 0.8, gamma 0.4
% 2) LML: alpha 0.6, delta 1.0, gamma 1.0


%% Load data, define parameters for the different utility functions
[startDir,~] = fileparts(mfilename('fullpath'));
load(fullfile(startDir,'data','allData.mat')); 
n_agents = 2;
alphas = [0.6];
deltas_cpt = [0.8];
gammas_cpt = [0.4];
deltas_lml = [1.0];
gammas_lml = [1.0];

betas = [exp(-1)];

subjList = [1];
Choice_synth = {};

counter = 0;
ii = 1;

%% Compute choices for prospect theory with different alphas, deltas and gammas
%the function "computeChoicesModelRecovery.m" returns choice probabilities 
%and synthetic choices probabilistically realized on the basis of the 
%choice probabilities, for both sessions
for a = 1:numel(alphas)
    for d = 1:numel(deltas_cpt)
        for g = 1:numel(gammas_cpt)
            for i = 1:n_agents
                counter = counter + 1;
                alpha = alphas(a);
                delta = deltas_cpt(d);
                gamma = gammas_cpt(g);

                fprintf('%d) PT: alpha %.1f, delta %.1f, gamma %.1f\n',counter,alpha,delta, gamma);

                [CP_left_PT,choice_left_PT] = ...
                    computeChoicesModelRecovery('PT',[alpha,delta,gamma,betas],subjList,...
                    {maxA,minA,p_maxA,maxB,minB,p_maxB});

                Choice_synth{i,ii} = choice_left_PT{:};
            end
            ii = ii+1;
        end
    end
end

%% Compute choices for LML model with different alphas, deltas and gammas
%the function "computeChoicesModelRecovery.m" returns choice probabilities 
%and synthetic choices probabilistically realized on the basis of the 
%choice probabilities, for both sessions
for a = 1:numel(alphas)
    for d = 1:numel(deltas_lml)
        for g = 1:numel(gammas_lml)
            for i = 1:n_agents
                counter = counter + 1;
                alpha = alphas(a);
                delta = deltas_lml(d);
                gamma = gammas_lml(g);

                fprintf('%d) LML: alpha %.1f, delta %.1f, gamma %.1f\n',counter,alpha,delta, gamma);

                [CP_left_lml,choice_left_lml] = ...
                    computeChoicesModelRecovery('LML',[alpha,delta,gamma,betas],subjList,...
                    {maxA,minA,p_maxA,maxB,minB,p_maxB});

                Choice_synth{i,ii} = choice_left_lml{:};
            end
            ii = ii+1;
        end
    end
end

%% Save synthetic data file 
%Data file with 2 different choices, but otherwise 2 times the same
%information from subject 1 (gambles presented to subject 1 and derived
%differences in utility)
for i = 1:numel(Choice_synth)
    maxA = maxA;
    minA = minA;
    xA   = xA;
    maxB = maxB;
    minB = minB;
    xB   = xB;

end
Choice = Choice_synth;

save(fullfile(startDir,'data','allData_synth_modelRecov.mat'),...
    'maxA', 'minA','xA', 'maxB', 'minB','xB', 'Choice');