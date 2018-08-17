function [PS,BS,CS]  = BL_Gaussian(seqoi, sig, order)

% This function creates Bayesian and predictive surprise regressors for a 
% binary input sequence based on the sequential updating of a zeroth-order
% Markov model= Bernoulli distribution Bayesian learner based on the 
% non-conjugate Gaussian distribution.
%
%   Input
%       seqoi      : sequence of interest, consisting of 0 and 1
%       sig        : assumed autoregressive process innovation variance (= "learning rate")
%       order      : zeroth or first order markov chain assumed
%
%   Output
%       BS         : Bayesian             surprise regressor of length n
%       PS         : Predictive           surprise regressor of length n
%       CS         : Confidence-corrected surprise regressor of length n
%
% Copyright (C) Kathrin Tertel
% -------------------------------------------------------------------------
T = length(seqoi);
if order==0                                                                 
    o_t=seqoi; % for zeroth-order, take sequence as is
else % for first-order, page 21 equation (30)
    o_t     = NaN(size(seqoi,1),1); % initialize stim change vector (called d_t in documentation)
    o_t(1)  = 0; % first stim is a change per definition
    T       = size(o_t,1); % length of sequence
    for t=2:T
        if seqoi(t-1) == seqoi(t) 
            o_t(t) = 1;
        else 
            o_t(t) = 0;
        end
    end
end


% unobservable random variable space initialization, page 44 equation (67)
s_min   = -5                                                                ; % minimum
s_max   =  5                                                                ; % maximum
s_res   = 7e1                                                               ; % resolution
s_sup   = linspace(s_min,s_max,s_res)                                       ; % support

% observed random variable space initialization
o_res   = 2                                                                 ; % resolution
o_sup   = [0 1]                                                             ; % support

% quantities of interest array initialization
p_filt          = NaN(s_res,T);
s_filt          = NaN(1,T);  
p_ott_stt_st    = NaN(o_res,s_res,s_res,T);
joint_naive     = NaN(o_res,s_res,s_res);
p_filt_naive    = NaN(1,s_res);

BS              = NaN(1,T);
PS              = NaN(1,T);
CS              = NaN(1,T);

% set surprise at t=1 to 0
BS(1)           = 0;
PS(1)           = 0;
CS(1)           = 0;

% initialization - zero precision Gaussian 
for i = 1:s_res
    p_filt(i,1) = 1/s_res;
end

% cycle over time points
for t = 2:T

    % inform user
    fprintf('Filtering t = %d of T = %d\n',t,T)
    
    % evaluate joint distribution p(o_t+1,s_t+1,s_t|o_1:t) (page 45, equation (69))
    for i = 1:o_res
        for j = 1:s_res
            for k = 1:s_res
                % page 45, equation (70)
                p_ott_stt_st(i,j,k,t) = binopdf(o_sup(i),1,1/(1+exp(-s_sup(j))))*normpdf(s_sup(j),s_sup(k),sig)*p_filt(k,t-1);
                % page 45, equation (75)
                joint_naive(i,j,k)    = binopdf(o_sup(i),1,1/(1+exp(-s_sup(j))))*normpdf(s_sup(j),s_sup(k),sig)*p_filt(k,1);
            end
        end
    end
    
    % evaluate unobserved variable distribution of interst  p(o_t+1 = o_t(t),s_t+1,s_t|o_1:t)
    p_ott_obs_stt_st = squeeze(p_ott_stt_st(o_t(t)+1,:,:,t));
    joint_naive_obs  = squeeze(joint_naive(o_t(t)+1,:,:));
    
    % integrate over s_t
    p_ott_obs_stt   = sum(p_ott_obs_stt_st,2);
    joint_naive_obs_int = sum(joint_naive_obs,2);
    
    % evaluate posterior 
    p_filt(:,t)     = p_ott_obs_stt/sum(p_ott_obs_stt(:)); % page 20 equation (26)
    p_filt_naive(:) = joint_naive_obs_int/sum(joint_naive_obs_int(:)); % page 45 equation (45)

    % evaluate posterior expectation
    s_filt(t)       = p_filt(:,t)'*s_sup';

    
    % evaluate surprises (page 21 equation (27))
    % evaluate predictive surprise 
    PS(t) = -log(sum(sum(p_ott_stt_st(o_t(t)+1,:,:,t),2),3)/sum(sum(sum(p_ott_stt_st(:,:,:,t),1),2),3));
    
    % evaluate Bayesian suprise - KL divergence between prior and posterior
    BS(t) = 0;
    for i = 1:s_res
        BS(t) = BS(t) + p_filt(i,t-1)*log(p_filt(i,t-1)/p_filt(i,t));
    end

    % evaluate confidence-corrected suprise - KL divergence between prior and naive posterior
    CS(t) = 0;
    for i = 1:s_res
        CS(t) = CS(t) + p_filt(i,t-1)*log(p_filt(i,t-1)/p_filt_naive(i));
    end 
end

end