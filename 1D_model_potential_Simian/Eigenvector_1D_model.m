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
state_start=9;
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

Ngauss=5;
for i=1:Ngauss
    C_g(i)=rand*100;
    std_g(i) = rand*25;
end
amp = 1.;

%% Initialize
initial_agent_pos=state_start;
T = 10^7;
num_windows = 1;
t = 0.01; % time steps
nodes = linspace(1,N,N);

%%
Record_states = zeros(num_windows, T);
Record_states(1,1) = initial_agent_pos;
x_rec = zeros(num_windows, T);
x_rec(1,1) = xx(Record_states(1,1));

pi_biased = zeros(num_windows, N);

K_biased = zeros(size(K));
mfpt = inf;
bias_opt = zeros(N,1);
t = .001;
for Nopt=1:1000
    for i=1:Ngauss
        C_g(i)=rand*100;
        std_g(i) = rand*25;
    end
    bias=zeros(N,1);
    for i = 1:1:Ngauss
        bias(1:N) = bias(1:N) + amp * exp(-(xx' - C_g(i)).^2/(2*std_g(i)^2));
    end
    for i = 1:N-1
        u_ij=bias(i+1)-bias(i);
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
    [mfpts] = mfpt_calc(pi_biased,K_biased);
    mfpt_mid = mfpts(state_start, state_end);
    M_t = expm(K_biased*t); % transition matrix.
    Mmfpt=Markov_mfpt_calc(pi_biased',M_t);
    mfpt_mid = Mmfpt(state_start, state_end)*t;

    %this line we update the optimized mfpt, and the bias along with it.
    if mfpt_mid < mfpt
        mfpt = mfpt_mid;
        bias_opt = bias;
    end
end
figure
plot(xx,F,'b', 'LineWidth', 2)
hold on
plot(xx,bias_opt, 'r', 'LineWidth', 2)
plot(xx,bias_opt+F'-min(bias_opt+F'), 'g--', 'LineWidth', 2)

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
t = .1; % time lag
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
