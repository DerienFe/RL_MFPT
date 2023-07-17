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

get_random_guess; % initial guess from 100 random bias attempts
%%%
% a=v;
% for is=state_start:1:state_end
%     v0=a;
%     %calculating  the optimal bias
%     %[a,b,c]= fmincon(@(v) min_reltime(v,K,N,KbT),v,[],[],[],[],LB,UB); 
%     [a,b,c]=fminsearch(@(v) min_mfpt(v,K,Ngauss,amp,kT,t,is,state_end),v0);%,optimset('MaxFunEvals',10000));
%     pos_bias{is}=a;
% end
%save pos_bias.mat pos_bias
load pos_bias.mat
%%
%%%% Initialize adaptive sims
Nsim=5;
initial_agent_pos=state_start;
T = 10^7;
nodes = linspace(1,N,N);
Record_states = zeros(Nsim, T);
Record_states(1,1) = initial_agent_pos;
sim_incr=1000;
for isim=1:Nsim
    Record_states(isim,1) = initial_agent_pos;
    current_pos=initial_agent_pos;
    not_reached=true;
    tot_time=0;
    trajectory=zeros(1, T);
    while not_reached
          v=pos_bias{current_pos};
          get_biased_M; %using v as the bias, and t as time steps
          % propagate the biased simulations for sim_incr steps
          [steps,trajectory,not_reached] = propagate_N_steps(M_t,sim_incr,nodes,current_pos,state_end,trajectory,tot_time,not_reached);
          tot_time=tot_time+steps;
          current_pos=trajectory(tot_time);
    end
    mfpt_adaptive(isim)=tot_time
end
average_mfpt_stochastic=mean(mfpt_adaptive)*t
error_mfpt_stochastic=std(mfpt_adaptive)*t
%%
%% using a single bias across the full sims
for i=1:5:70
v=pos_bias{i};
C_g(1:Ngauss) = v(1:Ngauss);
std_g(1:Ngauss) = v(Ngauss+1:2*Ngauss);
bias=zeros(N,1);
for i = 1:1:Ngauss
    bias(1:N) = bias(1:N) + amp * exp(-(xx' - C_g(i)).^2/(2*std_g(i)^2));
end
figure
plot(xx,F,'b', 'LineWidth', 2)
hold on
plot(xx,bias, 'r', 'LineWidth', 2)
plot(xx,bias+F'-min(bias_opt+F'), 'g--', 'LineWidth', 2)
end
%%%
%% generating biased stochastic trajectory
for i = 1:N-1
    u_ij=bias_opt(i+1)-bias_opt(i);
    K_biased(i,i+1)=K(i,i+1)*exp(-u_ij/2/kT);
    K_biased(i+1,i)=K(i+1,i)*exp(u_ij/2/kT);
    K_biased(i,i)=0;
end
K_biased(N,N)=0;
%% Normalizing the rate matrix.
for i = 1:N
    f=sum(K_biased(i,:));
    K_biased(i,i) =  -f;
end

%% biased MFPTs
[pi_biased, ~] = compute_free_energy(K_biased', kT);
plot(xx,-log(pi_biased)*kT-min(-log(pi_biased)*kT), 'c--', 'LineWidth', 2)

T = 10^7;
t = .01; % time lag
Nsim=10;
for i = 1:1:Nsim
    [Record_states(i,:),  step(i)] = explore_the_network(t,K_biased,bias_opt,xx,kT,T,cutoff,nodes,state_start,state_end);
end
average_mfpt_stochastic=mean(step)*t
error_mfpt_stochastic=std(step)*t
mmm=mfpt_calc(pi_biased,K_biased');
mfpt_from_rates=mmm(state_start, state_end)
M_t = expm(K_biased*t); % transition matrix.
Mmfpt=Markov_mfpt_calc(pi_biased',M_t);
mfpt_from_Markov=Mmfpt(state_start, state_end)*t

