function [PS,BS,CS] = BL_Betabern(seqoi, tau, order)

% This function creates Bayesian and predictive surprise regressors for a 
% binary input sequence based on the sequential updating of a zeroth- or 
% first-order Markov model= Bernoulli distribution Bayesian learner based 
% on the conjugate Beta-distribution.
%
%   Input
%       seqoi       : binary sequence of interest of length n
%       tau         : exponential weighting constant
%       order       : zeroth- or first-order markov chain with symmetrical
%                     transition probabilities assumed
%
%   Output
%       PS          : Predictive           surprise regressor of length n
%       BS          : Bayesian             surprise regressor of length n
%       CS          : Confidence-corrected surprise regressor of length n
%
% Copyright (C) Kathrin Tertel
% -------------------------------------------------------------------------
if order==0                                                                 
    o_t=seqoi; % for zeroth-order, take sequence as is
else % for first-order, page 19 equation (21)
    % initialize stim change vector (called d_t in documentation)
    o_t     = NaN(size(seqoi,1),1); 
    o_t(1)  = 0; % first stim is a change per definition
    T       = size(o_t,1); % length of sequence
    for t=2:T
        if seqoi(t-1) == seqoi(t) 
            o_t(t) = 0;
        else 
            o_t(t) = 1;
        end
    end
end

% define state space
S  = unique(o_t);

% initialize Bayesian and predictive surprise regressors
PS  = zeros(1,length(o_t));
BS  = zeros(1,length(o_t));
CS = zeros(1,length(o_t));
% sequential Bayesian learning
% -------------------------------------------------------------------------
% parameter flags for weighting/initial prior over s_t
alpha_beta_flag         = [1 1];

% cycle over trials
for t = 1:length(o_t)    
    % create exponential downweighting weights - weight for flag t => 1 %
    % page 15 equation (15)
    w_pri          = repmat(exp(-tau.*[t-1:-1:0])',1,2);
        
    % record prior alpha
    alpha_beta_pri = sum(w_pri.*alpha_beta_flag,1); % page 15, equation (13)
    
    % if the state at t is 0 add one to alpha, if the state at t is 1 add one to beta 
    if o_t(t) == S(1) 
        alpha_beta_flag = [alpha_beta_flag; 1 0];
        alpha_beta_naive_flag = [1 1; 1 0];
    elseif o_t(t) == S(2) 
        alpha_beta_flag = [alpha_beta_flag; 0 1];
        alpha_beta_naive_flag = [1 1; 0 1];
    end
    
    % create exponential downweighting weights - weight for flag t => 1
    % page 15 equation (15)
    w_pos           = repmat(exp(-tau.*[t:-1:0])',1,2);
 
    % record posterior alpha/beta parameters
    alpha_beta_pos  = sum(w_pos.*alpha_beta_flag,1); % page 15, equation (13)/(15)
    alpha_beta_pos_naive   = sum(alpha_beta_naive_flag,1);
    
    % evaluate surprises: page (15) equation (14)
    % evaluate the negative log probability of the current observation
    PS(t)  = -log(((alpha_beta_pri(1)/sum(alpha_beta_pri,2))^(o_t(t)==S(1))*(1 - (alpha_beta_pri(1)/sum(alpha_beta_pri,2)))^(o_t(t)==S(2)))); 
   
    % evaluate the KL divergence between prior and the currently relevant (= updated) posterior relevant beta distributions
    BS(t) = spm_kl_dirichlet(alpha_beta_pri, alpha_beta_pos); 

    % evaluate the KL divergence between prior and naive posterior
    CS(t) = spm_kl_dirichlet(alpha_beta_pri, alpha_beta_pos_naive);
    
end
end
