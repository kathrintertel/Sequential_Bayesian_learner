function [Q] = sample_hhmm(T,changes,bigchange)

% This function implements an ancestral sampling scheme for an hierarchical
% hidden Markov model.
%
%   Input
%       T:          sequence length, natural number
%       changes:    change probabilities for the two volatility
%                   conditions, 2D real vector
%       bigchange:  regime change probability
%   Output
%                   Q : state sequence for first and second level
%
%                   
% Copyright (C) Kathrin Tertel
% -------------------------------------------------------------------------
clc
close all
do_plot = 0;

min_first_level_sequence = 30; % minimal sequence length without regime change

% Model Specification
% -------------------------------------------------------------------------

% hierarchy depth
D       = 2;

% create 0th level transition probability matrix
A{1}{1} = [0 1; 1 0];


A{2}{1} = [1-changes(1)-bigchange changes(1) - bigchange bigchange*2; changes(1) - bigchange  1-changes(1)-bigchange bigchange*2; 0 0 0];
A{2}{2} = [1-changes(2)-bigchange changes(2) - bigchange bigchange*2; changes(2) - bigchange  1-changes(2)-bigchange bigchange*2; 0 0 0];


% create 0th level initial distribution 
P{1}{1} = [1/2 1/2];

% create 1st level initial distributions
P{2}{1} = [.5 .5 0];
P{2}{2} = [.5 .5 0];


% initialize the state sampling array, number of state levels x sequence
% length
Q       = NaN(D,T);

% Model Sampling
% -------------------------------------------------------------------------

% sample the first first level state based on the 0th level initial
% distribution 
Q(1,1) = find(logical(mnrnd(1, P{1}{1})));

% sample the second level state based on the current 1st level initial
% distribution
Q(2,1) = find(logical(mnrnd(1, P{2}{Q(1,1)})));

last_change = 0;

for t = 2:T
    
    % sample the second level states according to the current first level
    % state's transition probability matrix
    q2 = find(logical(mnrnd(1, A{2}{Q(1,t-1)}(Q(2,t-1),:))));
    
    % if the second level state is the end state, resample the first level
    % state and second level state until a non-end state is reached
    while q2 == 3
       if t > 2
            if t - last_change < min_first_level_sequence
                Q(1,t-1) = Q(1,t-2);
            else   
                % sample the second level state based on the current 0th level 
                % transition probability matrix
                Q(1,t-1) = find(logical(mnrnd(1, A{1}{1}(Q(1,t-2),:))));
                if Q(1,t-1) ~= Q(1,t-2);
                    last_change = t;
                end
            end
            % resample the second level 
            q2     = find(logical(mnrnd(1, A{2}{Q(1,t-1)}(Q(2,t-1),:))));
        else
            % sample the first first level state based on the 0th level initial
            % distribution 
            Q(1,1) = find(logical(mnrnd(1, P{1}{1})));

            % sample the second level state based on the current 1st level initial
            % distribution
            q2 = find(logical(mnrnd(1, P{2}{Q(1,1)})));
        end
    end
    
    % if the end state is not reached, update the first level state and
    % re-sample based on its transition probability matrix on the next
    % iteration of the for loop
    Q(1,t) = Q(1,t-1);
    
    % save the sample second level state
    Q(2,t) = q2;
    
       
end

% Plot the sample sequence of first and second level states
% -------------------------------------------------------------------------
if do_plot
    h = figure;
    set(h, 'color', [1 1 1])

    subplot(2,1,1)
    plot([1:T], Q(1,:), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 2)
    ylim([.5 2.5]) 
    set(gca, 'ytick', [1 2], 'yticklabel', {'High Volatility', 'Low Volatility'})
    xlabel('t')
    title('First Level State')

    subplot(2,1,2)
    plot([1:T], Q(2,:), 'ko-', 'MarkerFaceColor', 'k', 'MarkerSize', 2)
    ylim([.5 2.5]) 
    set(gca, 'ytick', [1 2], 'yticklabel', {'A1', 'A2'})
    xlabel('t')
    title('Second Level State')
    maximize(h)
    close(h)
    %saveas(h, [cd '\Figures\HHMM_Sample.bmp'])
end


