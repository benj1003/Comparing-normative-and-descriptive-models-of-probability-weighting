function [CP,choice] = computeChoicesModelRecovery(utFunc,params,subjList,gambs)
%% computeChoicesModelRecovery
% function loops over (additive) session and 
% 1. computes the utility for the gambles (and their difference) given the 
% utility function and the relevant parameter values (calling 
% computeExpectedUtility.m 
% 2. computes choice probabilities (for choosing gamble on the left side)
% given a beta value
% 3. computes choices (0 = right, 1 = left) probabilistically realised on
% the basis of the choice probability (calling computeChoiceRealisation.m)

    CP = {}; %cell array to be filled with choice probabilities (for 
    %left gamble) for every subject, every trial, additive session
    choice = {}; %probabilistically realised choices (left=1, right=0)
    %based on choices probabilities, for every subject, every trial,
    %additive session
    pa =  gambs{3}; %probability of gamble outcomes 
    pb = gambs{6};
    for i = subjList 
        %Prospect theory
        if strcmpi(utFunc,'PT')
            alpha = params(1);
            delta = params(2);
            gamma = params(3);
            beta = params(4);
            
            CP{i} = nan(size(gambs{1}));

            for t = 1:numel(gambs{1})
                Ut_PT_left = computeExpectedUtility({gambs{1}(t),gambs{2}(t)},alpha,delta,gamma,[pa(t)]);
                Ut_PT_right = computeExpectedUtility({gambs{4}(t),gambs{5}(t)},alpha,delta,gamma,[pb(t)]);
                CP{i}(t) = 1/(1+exp(-((Ut_PT_left-Ut_PT_right)*beta)));
                choice{i}(t) = computeChoiceRealisation(CP{i}(t));
            end
        %LML
        elseif strcmpi(utFunc,'LML')
            alpha = params(1);
            delta = params(2);
            gamma = params(3);
            beta = params(4);
            
            CP{i} = nan(size(gambs{1}));

            for t = 1:numel(gambs{1})
                Ut_PT_left = computeExpectedUtility_lml({gambs{1}(t),gambs{2}(t)},t,alpha,delta,gamma,[pa(t)]);
                Ut_PT_right = computeExpectedUtility_lml({gambs{4}(t),gambs{5}(t)},t,alpha,delta,gamma,[pb(t)]);
                CP{i}(t) = 1/(1+exp(-((Ut_PT_left-Ut_PT_right)*beta)));
                choice{i}(t) = computeChoiceRealisation(CP{i}(t));
            end
        end
    end
end