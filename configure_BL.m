function configure_BL
% This function creates a structure T to be used for the Bayesian learner
% functions BL_Betabern, BL_Betabern_TP, and BL_Gaussian.
% 
% A two-item input sequence can be sampled with the hierarchical hidden
% markov model using the function sample_hhmm.
%
% Copyright (C) Kathrin Tertel
% -------------------------------------------------------------------------


C               = [];           % intitialize configuration

C.T             = 200;          % sequence lenghth
C.changes       = [.25,.75];    % transitional probabilities
C.bigchange     = .01;          % regime-change-probability
C.sequence      = sample_hhmm(C.T,C.changes,C.bigchange); % generate sequence with hierarchical Markov-chain

% model parameters
C.forgetting    = 0.14;         % forgetting via exponential downweighting for Beta-Bernoulli models
C.variance.high = 2.5;          % assumed volatility for Gaussian models (high)
C.variance.low  = 0.1;          % assumed volatility for Gaussian models (low)
plot_BL(C)

end