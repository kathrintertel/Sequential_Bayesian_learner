# Sequential_Bayesian_learner

This code is written for Matlab and may be used to create Bayesian learner models according to a Beta-Bernoulli or Gaussian random walk scheme and 
compute predictive, Bayesian, and confidence-corrected surprise.

Function configure_BL depends on sample_hhmm as well as plot_BL and allows the user to choose settings for the computed Bayesian learner models.
In plot_BL, surprise regressors are computed according to the settings chosen in configure_BL and consequently plotted. 

plot_BL depends on BL_Betabern, BL_Betabern_TP and BL_Gaussian to calculate surprise regressors.
