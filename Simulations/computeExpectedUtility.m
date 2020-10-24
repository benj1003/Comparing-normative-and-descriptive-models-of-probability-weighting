function [ eu ] = computeExpectedUtility(outcome,alpha,delta,gamma,probVals)
%% computeExpectedUtility
% This takes two outcome spaces outcome{1} and outcome{2} and computes
% expected utility according to prospect theory givne the input exponents,
% loss aversion parameters (lambda), and probability values (probVals), via
% a weighted sum, as per cumulative prospect theory.

x1=double(outcome{1});% outcome 1
x2=double(outcome{2});% outcome 2
p1=double(probVals(1));% probability of outcome 1

try
    p2=1-p1;% probability of outcome 2
    if p1+p2 > 1 % ensure probabilites do not exceed 1
        disp('Error: probabilities sum > 1')
    end
end

%%Outcome 1 
%Values 
v1=nan(size(x1));
v1=x1.^alpha;

%weights
w1=nan(size(p1));
tmp=delta*(p1^gamma)+(1-p1)^gamma;
w1=(delta*(p1^gamma)) / tmp;

eu = (v1*w1);

try
    
    %% Values for outcome 2
    v2=nan(size(x2));
    v2=x2.^alpha;
    %v2(outcome{2}>=0)=outcome{2}(outcome{2}>=0).^alpha;%gains
    
    %weights
    w2=nan(size(p1));
    tmp=delta*(p2.^gamma)+(1-p2).^gamma;
    w2=(delta*(p2.^gamma)) / tmp;
    
    %% Weighted sum
    w1 = w1/(w1+w2);
    w2 = w2/(w1+w2);
    eu = (v1*w1) + (v2*w2);
    
end


end

