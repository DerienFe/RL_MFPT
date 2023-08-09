%%
clear all
clc

% parameter initialization
N=100;
kT=0.596;
Ngauss=20;
amp = 1.;
ts = 0.01;

%initialize the environment
env = MFPT_env();

%do the rest of RL training in the RL designer app.
%by typeing: reinforcementLearningDesigner


%testing the env.
rng(0);
InitialObs = reset(env);



