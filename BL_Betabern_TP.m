function [PS1,BS1,CS1]  = BL_Betabern_TP(seqoi, tau)

% This function creates a Bayesian surprise regressor for a binary input
% sequence based on the sequential updating of an unconstrained Bayesian
% Markov-chain using the conjugate Beta-distribution.
%
%   Inputs
%       seqoi       : binary sequence of interest of length n
%       tau         : exponential weighting constant
%
%   Outputs
%       PS1         : Predictive           surprise regressor of length n
%       BS1         : Bayesian             surprise regressor of length n
%       CS1         : Conficence-corrected surprise regressor of length n
%
% Copyright (C) Kathrin Tertel
% -------------------------------------------------------------------------
% define state space
S  = unique(seqoi); 

% initialize Bayesian and predictive surprise regressors
PS1 = zeros(1,length(seqoi));
BS1 = zeros(1,length(seqoi));
CS1 = zeros(1,length(seqoi));

% sequential Bayesian learning
% -------------------------------------------------------------------------
% initialize flags to zero array
alpha_flag          = zeros(2,2,length(seqoi));
alpha_flag_naive    = zeros(2,2,2);

% initial priors over rows of the 2 x 2 transition probability matrix
alpha_flag(:,:,1)         = ones(2,2);
alpha_flag_naive(:,:,1)   = ones(2,2);

% cycle over trials
for t = 2:length(seqoi)  
    % create exponential downweighting weights - weight for flag t => 1 %
    % page 17 equation (19)
    w_pri          = exp(-tau.*[t-2:-1:0]); 
    alpha_flag_naive(:,:,2)   = zeros(2,2); % at each t, set flag of naive posterior back to zero
    % weight alpha flags
    for k = 1:length(w_pri)
        w_alpha_flag(:,:,k) = alpha_flag(:,:,k)*w_pri(k);
    end
       
    % record prior alpha
    alpha_pri      = sum(w_alpha_flag(:,:,1:t-1),3);
  

    % cycle over states at t-1
    for i = 1:length(S)
        
        % cycle over states at t 
        for j = 1:length(S)
        
            % if the state at t is j and if the state at t-1 is i set flag at position (i,j)
            if seqoi(t) == S(j) && seqoi(t-1) == S(i)
                alpha_flag(i,j,t) = 1; % page 17 equation (17)
                alpha_flag_naive(i,j,2)   = 1;
            end
            
        end
    end
    
    % create exponential downweighting weights - weight for flag t => 1
    % page 17 equation (19)
    w_pos           = exp(-tau.*[t-1:-1:0]);
    
    % weight alpha flags
    for k = 1:length(w_pos)
        w_alpha_flag(:,:,k) = alpha_flag(:,:,k)*w_pos(k);
    end
    
    % record posterior alpha
    alpha_pos       = sum(w_alpha_flag(:,:,1:t),3);
    % record posterior naive alpha unweighted
    alpha_naive     = sum(alpha_flag_naive,3);
    
    % cycle over states at t-1 
    % Evaluate surprises (page 17 equation (18))
    for i = 1:length(S)   
        if seqoi(t-1) == S(i)
            % evaluate Bayesian Surprise (the KL divergence between the 
            % currently relevant (= updated) prior and posterior row 
            % Dirichlet distributions) 
            BS1(t) = spm_kl_dirichlet(alpha_pri(i,:), alpha_pos(i,:));
            % evaluate Confidence-corrected Surprise (the KL divergence 
            % between the currently relevant (= updated) prior and naive 
            % posterior row Dirichlet distributions )
            CS1(t) = spm_kl_dirichlet(alpha_pri(i,:), alpha_naive(i,:)); 
        end        
        % cycle over states at t 
        for j = 1:length(S)    
            % if the state at t is j and if the state at t-1 is i evaluate
            % the posterior expectation of the transition probability i -> j
            if seqoi(t) == S(j) && seqoi(t-1) == S(i)
                % evaluate Predictive Surprise (the negative log 
                % probability of the current observation)
                PS1(t) = -log(alpha_pri(i,j)./sum(alpha_pri(i,:),2)); 
            end  
        end
    end 
     
end