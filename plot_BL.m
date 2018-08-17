function plot_BL(C)

% This function runs several Bayesian Learner functions to generate 
% surprise regressors as discussed in Tertel et al. (2017) and plots them.
% To run this function properly, the TAPAS toolbox is needed
% (http://www.translationalneuromodeling.org/tapas/).
%
%   Input (can be configured using configure_BL.m)
%       
%       C:                  a structure specifying several sequence- and
%                           BL-properties, such as:
%
%       C.T:                sequence length, natural number; set to 200 in Tertel
%                           et al. (2018).
%       C.changes:          change probabilities for the two volatility
%                           conditions, 2D real vector; set to [.25,.75] in Tertel
%                           et al. (2018).
%       C.bigchange:        regime change probability; set to .01 in Tertel
%                           et al. (2018).
%       C.sequence          2xT sequence vector generated with 
%                           sample_hhmm(C.T,C.changes,C.bigchange). First row 
%                           codes regime (1 = slow, 2 = fast regime), second
%                           row codes stimuli (1 = low, 2 = high).
%       C.forgetting:       exponential forgetting parameter, set to 0.14 in 
%                           Tertel et al. (2018).
%       C.variance.high     high volatiliy parameter, set to 2.5 in 
%                           Tertel et al. (2018).
%       C.variance.low      low volatiliy parameter, set to 0.1 in 
%                           Tertel et al. (2018).
%      
%                   
% Copyright (C) Kathrin Tertel 
% Documented in: Tertel, K., Blankenburg, F., Ostwald, D. (2018).
% Probabilistic Computational Models for Neural Surprise Signals.
% -------------------------------------------------------------------------
clc

fid = 'surprise_regressors';                                                % file identifier for saving sequence properties and plots

calc_BLs = 0;                                                               % 1: surprise regressors will be calculated
plot_BLs = 1;                                                               % 1: surprise regressors will be plotted

% specify directory to save to
savedir = 'C:\';                                                

if calc_BLs == 1
    load([savedir fid '.mat'],'C')
    seqoi           = C.sequence(2,:);                                      % extract stimulus vector
    seqoi           = seqoi(:)-1;                                           % transform stimulus vector to 0 and 1

    [PS0N, BS0N, CS0N]       = BL_Betabern(seqoi,0,0);                      % BB SP no forgetting
    [PS0F, BS0F, CS0F]       = BL_Betabern(seqoi,C.forgetting,0);           % BB SP with forgetting
    
    [PS1N, BS1N, CS1N]       = BL_Betabern(seqoi,0,1);                      % BB AP no forgetting
    [PS1F, BS1F, CS1F]       = BL_Betabern(seqoi,C.forgetting,1);           % BB AP with forgetting    
    
    [PS1TN, BS1TN, CS1TN]    = BL_Betabern_TP(seqoi,0);                     % BB TP no forgetting
    [PS1TF, BS1TF, CS1TF]    = BL_Betabern_TP(seqoi,C.forgetting);          % BB TP with forgetting
    
    [PS0H, BS0H, CS0H]       = BL_Gaussian(seqoi,C.variance.high,0);        % GRW SP high volatility
    [PS0L, BS0L, CS0L]       = BL_Gaussian(seqoi,C.variance.low,0);         % GRW SP low volatility    
    
    [PS1H, BS1H, CS1H]       = BL_Gaussian(seqoi,C.variance.high,1);        % GRW AP high volatility
    [PS1L, BS1L, CS1L]       = BL_Gaussian(seqoi,C.variance.low,1);         % GRW AP low volatility

    models          = [PS0N',  PS0F',  BS0N',  BS0F',  CS0N',  CS0F', ...
                       PS1N',  PS1F',  BS1N',  BS1F',  CS1N',  CS1F',...
                       PS1TN', PS1TF', BS1TN', BS1TF', CS1TN', CS1TF',...
                       PS0H',  PS0L',  BS0H',  BS0L',  CS0H',  CS0L',...
                       PS1H',  PS1L',  BS1H',  BS1L',  CS1H',  CS1L'];

    C.modlabs       = {'PS0N',  'PS0F',  'BS0N',  'BS0F',  'CS0N',  'CS0F', ...
                       'PS1N',  'PS1F',  'BS1N',  'BS1F',  'CS1N',  'CS1F',...
                       'PS1TN', 'PS1TF', 'BS1TN', 'BS1TF', 'CS1TN', 'CS1TF',...
                       'PS0H',  'PS0L',  'BS0H',  'BS0L',  'CS0H',  'CS0L',...
                       'PS1H',  'PS1L',  'BS1H',  'BS1L',  'CS1H',  'CS1L'};

    save([savedir fid '.mat'],'models','C')
end    

if plot_BLs     == 1
    load([savedir fid '.mat'],'models','C')

    seqoi       = C.sequence(2,:);                                                      
    seqoi       = seqoi(:)-1;

    cols        = [29,  90, 171;  % 1. blue
                  153,  39,  39;  % 3. red
                  140, 124,  34;  % 2. yellow                                  
                   33, 128,  65]; % 4. green

    cols        = cols/255;  

    for fig     = 1:5 % loop over different plots containing the stim sequence on top and 6 regressors below
        h       = figure;
        set(h, 'Color', [1 1 1])
        subplot(5,1,1)
        plot(seqoi(:), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 2)
        ylim([-1 2])
        set(gca, 'ytick', [0 1], 'yticklabel', {'0', '1'}, 'FontSize', 10)
        
        id_mod_n = [(fig-1)*6+1;(fig-1)*6+3;(fig-1)*6+5]; % select regressors no forgetting/high volatility
        curr_mod_n = models(:,id_mod_n);
        id_mod_f = [(fig-1)*6+2;(fig-1)*6+4;(fig-1)*6+6]; % select regressors with forgetting/low volatility
        curr_mod_f = models(:,id_mod_f);

        markerline = {'-o', '-d'};
         
        subplot(5,1,2) % subplot no forgetting/high volatility
        plot(1:200, models(:,id_mod_n(1)), markerline{1}, 'Color', cols(1,:), 'MarkerFaceColor', cols(1,:), 'MarkerSize', 2) % plot PS   
        hold on
        plot(1:200, models(:,id_mod_n(2)), markerline{1}, 'Color', cols(2,:), 'MarkerFaceColor', cols(2,:), 'MarkerSize', 2) % plot BS  
        plot(1:200, models(:,id_mod_n(3)), markerline{1}, 'Color', cols(3,:), 'MarkerFaceColor', cols(3,:), 'MarkerSize', 2) % plot CS
        set(gca, 'xticklabel',[])
        hold off
        ylim([0 2.75])
        
        subplot(5,1,3) % subplot with forgetting/low volatility
        plot(1:200, models(:,id_mod_f(1)), markerline{2}, 'Color', cols(1,:), 'MarkerFaceColor', cols(1,:), 'MarkerSize', 3) % plot PS  
        hold on
        plot(1:200, models(:,id_mod_f(2)), markerline{2}, 'Color', cols(2,:), 'MarkerFaceColor', cols(2,:), 'MarkerSize', 3) % plot BS
        plot(1:200, models(:,id_mod_f(3)), markerline{2}, 'Color', cols(3,:), 'MarkerFaceColor', cols(3,:), 'MarkerSize', 3) % plot CS
        set(gca, 'xticklabel',[])
        %legend('boxoff')
        hold off
        ylim([0 3.5])
        
        norm_mod_n     = NaN(200,3); % compute normalized models                                                                     
        for m            = 1:3            
            m_mod        = mean(curr_mod_n(1:200,m));
            s_mod        = std(curr_mod_n(1:200,m),[],1);
            norm_mod_n(:,m)=(curr_mod_n(:,m)-m_mod)/s_mod;
        end  
        
        subplot(5,1,4) % plot normalized models 
        plot(1:200, norm_mod_n(:,1), markerline{1}, 'Color', cols(1,:), 'MarkerFaceColor', cols(1,:), 'MarkerSize', 2) % plot normalized PS
        hold on
        plot(1:200, norm_mod_n(:,2), markerline{1}, 'Color', cols(2,:), 'MarkerFaceColor', cols(2,:), 'MarkerSize', 2) % plot normalized BS
        plot(1:200, norm_mod_n(:,3), markerline{1}, 'Color', cols(3,:), 'MarkerFaceColor', cols(3,:), 'MarkerSize', 2) % plot normalized CS
        hold off 
        set(gca, 'xticklabel',[])
        box on)
        ylim([-2 3])
        
        norm_mod_f     = NaN(200,3); % compute normalized models
        for m            = 1:3 
            m_mod        = mean(curr_mod_f(1:200,m));
            s_mod        = std(curr_mod_f(1:200,m),[],1);
            norm_mod_f(:,m)=(curr_mod_f(:,m)-m_mod)/s_mod;
        end
        subplot(5,1,5)                                                      % plot normalized models
        plot(1:200, norm_mod_f(:,1), markerline{2}, 'Color', cols(1,:), 'MarkerFaceColor', cols(1,:), 'MarkerSize', 3) % plot normalized PS
        hold on
        plot(1:200, norm_mod_f(:,2), markerline{2}, 'Color', cols(2,:), 'MarkerFaceColor', cols(2,:), 'MarkerSize', 3) % plot normalized BS
        plot(1:200, norm_mod_f(:,3), markerline{2}, 'Color', cols(3,:), 'MarkerFaceColor', cols(3,:), 'MarkerSize', 3) % plot normalized CS  
        hold off
        box on
        
        saveas(h, [savedir fid '_newset_' num2str(fig) '.fig'])
        saveas(h, [savedir fid '_newset_' num2str(fig) '.pdf'])
    end
end
end
