%%
clear all
clc

N=100;
kT=0.596;

%% Create a trasition rate matrix

K = create_K_1D(N, kT);

%% obtain free energy term.

[eq, F, eigenvectors, eigenvalues, eigenvalues_sorted, index] = compute_free_energy(K',kT);
F=F-min(F);
[mfpts] = mfpt_calc(eq,K);
state_start=1;
state_end = 89;
mfpt0 = mfpts(state_start, state_end)
%% Kemeny constant check
kemeny=zeros(N,1);
for i=1:N
    for j=1:N
            kemeny(i)=kemeny(i)+mfpts(i,j)*eq(j);
    end
end
min(kemeny)
max(kemeny)
Kemeny_eig=-sum(1./eigenvalues_sorted(2:N))

%% cutoff for numerical exp issues
cutoff = 20;
xx = linspace(1,N,N);

% bias size and type
Ngauss=20;
amp = 1.;
t = 0.01; % time steps
%load pos_bias.mat
%%%% Initialize adaptive sims
Nsim=1;
initial_agent_pos=state_start;
T = 10^7;
nodes = linspace(1,N,N);
Record_states = zeros(Nsim, T);
Record_states(1,1) = initial_agent_pos;
sim_incr=1000;
for isim=1:Nsim
    current_pos=initial_agent_pos;
    not_reached=true;
    tot_time=0;
    trajectory=zeros(1, T);
    incr_step=0;
    get_random_guess; % initial guess from 100 random bias attempts
    v=bias_opt;
    %get_biased_M;
    M0=expm(K*t);
    while not_reached
          v=predict_bias(M0,v,cutoff,Ngauss,amp,kT,current_pos,state_end,tot_time);
          get_biased_M; %using v as the bias, and t as time steps
          % propagate the biased simulations for sim_incr steps
          [steps,trajectory,not_reached] = ...
           propagate_N_steps(M_t,sim_incr,nodes,current_pos,state_end,trajectory,tot_time,not_reached);
          tot_time=tot_time+steps;
          current_pos=trajectory(tot_time);
          incr_step=incr_step+1;
          datlength(incr_step)=steps;
          bias_traj{incr_step}=v;
          [prob_dist,M0] = DHAM_sym(trajectory, bias_traj, kT, N, datlength,cutoff,amp);
          pi_pred{incr_step}=prob_dist;
    end
    %[prob_dist,M0] = DHAM_unbias(trajectory, bias_traj, kT, N, datlength,cutoff,amp);
    Record_states(isim,1:tot_time) = trajectory(1:tot_time);
    mfpt_adaptive(isim)=tot_time
end
% average_mfpt_stochastic=mean(mfpt_adaptive)*t
% error_mfpt_stochastic=std(mfpt_adaptive)*t
%%
