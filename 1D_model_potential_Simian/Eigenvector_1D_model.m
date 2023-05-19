%%
clear all
clc

N=100;
n_states=N;
force_constant = 0.1;
kT=0.5981;

%% Create a trasition rate matrix

K = create_K_1D(N, kT);

%% obtain free energy term.

[pi, F, eigenvectors, eigenvalues, eigenvalues_sorted, index] = compute_free_energy(K, kT);
F=F-min(F);
[mfpts] = mfpt_calc(pi,K,N);
state_start=9;
state_end = 89;
mfpt0 = mfpts(state_start, state_end)

%% Umbrella sampling parameters.

num_windows = 1;
cutoff = 20;
lambda = 2;
xx = eigenvectors(:,index(lambda))'; % set the projection coordinate
xx = linspace(1,N,N);
x_sorted = sort(xx);
x_eq = linspace(x_sorted(1), x_sorted(end), num_windows)
x_j = xx';

Ngauss=5;
for i=1:Ngauss
    C_g(i)=rand*100;
    std_g(i) = rand*25;
end
amp = 1.;
F_bias = F;

%% Initialize

%initial_agent_pos = find(xx == min(xx),1);
initial_agent_pos=state_start;
T = 10^7;
t = 1.; % time steps
nodes = linspace(1,n_states,n_states);

%%
Record_states = zeros(num_windows, T);
Record_states(1,1) = initial_agent_pos;
x_rec = zeros(num_windows, T);
x_rec(1,1) = xx(Record_states(1,1));

pi_biased = zeros(num_windows, n_states);
bias_order_win = zeros(num_windows, n_states);

K_biased = zeros(size(K));
mfpt = inf;
bias_opt = zeros(N,1);

%try 1000 times to find the lowest mfpt.
%stored in mfpt. bias_opt.
for Nopt=1:1000 

    % initialize 5 gaussians
    for i=1:Ngauss
        C_g(i)=rand*100;
        std_g(i) = rand*25;
    end

    %calculate the total bias summing up all 5 gaussians.
    bias=zeros(N,1);
    for i = 1:1:Ngauss
        bias(1:N) = bias(1:N) + amp * exp(-(x_j - C_g(i)).^2/(2*std_g(i)^2));
    end

    %     plot(xx, bias)
    %     hold on

    %calculate the biased transition rate matrix.
    for i = 1:1:length(xx)  %note xx is just x-axis. [1,..., 100]
        x_i = xx(i); %1 to 100 in shape [100,1]
        x_j = xx'; %1 to 100 in shape [100, 1]
        u_ij = bias - bias(x_i); %what's this?
        u_ij(u_ij>cutoff) = cutoff;
        u_ij(u_ij<-cutoff) = -cutoff;

        KK = K(i,:);

        KK = KK'.*exp(u_ij./(2*kT)); %the same.
        K_biased(i,:) = KK';
        K_biased(i,i) = 0;

    end

    %% Normalizing the transition rate matrix.

    for i = 1:1:size(K_biased,1)

        f=sum(K_biased(:,i));
        K_biased(i,i) =  -f;

    end

    %% biased MFPTs
    [pi_biased, ~] = compute_free_energy(K_biased, kT);
    [mfpts] = mfpt_calc(pi_biased,K_biased,N);
    mfpt_mid = mfpts(state_start, state_end);
    mfpt_mid;
    
    if mfpt_mid < mfpt
        mfpt = mfpt_mid;
        bias_opt = bias;
    end
end


%% Plot the free energy landscape and the opt bias potential.
figure
plot(xx,F,'b', 'LineWidth', 2)
hold on
plot(xx,bias_opt, 'r', 'LineWidth', 2)
%disp(mftp);
%disp('Hello');


%%%
%% Umbrella sampling

%here we use the optimal bias in the last Monte Carlo simulation.
%apply the opt bias, calculate K, MFPT.
for i = 1:1:length(xx)
    x_i = xx(i);
    x_j = xx';
    u_ij = bias_opt - bias_opt(x_i);
    u_ij(u_ij>cutoff) = cutoff;
    u_ij(u_ij<-cutoff) = -cutoff;

    KK = K(i,:);

    KK = KK'.*exp(u_ij./(2*kT));
    K_biased(i,:) = KK';
    K_biased(i,i) = 0;
end

%% Normalizing the transition rate matrix.

for i = 1:1:size(K_biased,1)
    f=sum(K_biased(:,i));
    K_biased(i,i) =  -f;
end

%% biased MFPTs
[pi_biased, ~] = compute_free_energy(K_biased, kT);


%what's different starts here.
count = 0;
T = 10^7;
t = .01; % time steps
Nsim=3;
for i = 1:1:Nsim
    [Record_states(i,:),  step(i)] = explore_the_network(t,K_biased,bias_opt,xx,kT,T,cutoff,nodes,state_start,state_end);
end
mean(step)/100
mfpt


%% Histogram of nodes visited.
figure
histogram(Record_states(:), 'Normalization','pdf')
ylabel('probability of visit')

%% Unbias with DHAM

[prob_dist] = DHAM_unbias(Record_states, x_eq, force_constant, kT, n_states, bias_order_win, cutoff);
free_energy = -kT.*log(prob_dist);

%% Probability distribution from DHAM

figure
hold on
plot(1:1:n_states,pi, 'LineWidth',2)
plot((1:n_states),prob_dist, 'LineWidth',2) % probability distribution
legend('Exact','DHAM')
ylabel('Probability')
xlabel(['Projection coordinate (node)']);
title('probabilty distribution')
box on

%% Free energy profile from DHAM

figure 
hold on
RMSE_free_energy = sqrt(mean ( ( free_energy - F).^2 ));
plot((1:n_states),F-min(F), 'LineWidth',2)
plot((1:n_states),free_energy - min(free_energy), 'LineWidth',2)
legend('Exact','DHAM')
title(['Free energy RMSE:', num2str(RMSE_free_energy)])
box on

%% Traceplot of nodes visited.

RR = Record_states';
figure
plot(RR(:))
xlabel('time')
ylabel('nodes')
ylim([0, n_states])
title('traceplot of nodes visited')

%% plot the biased probability distributions from each window. p_biased is obtained analytically

figure
hold on
plot(xx,pi,'-x')

for i = 1:1:num_windows
    
    plot(xx,pi_biased(i,:),'-x')
    
end

xlim([min(xx), max(xx)])
box on
xlabel('reaction coordinate')
ylabel('Biased probability')

%% Compute Biased Probability density as obtained empirically.

window = linspace(1,num_windows,num_windows);

figure
hold on
for i = 1:1:num_windows
    
    uv = unique( x_rec(window(i),:) );
    n  = histc(x_rec(window(i),:),uv);
    n = n./sum(n);
    
    plot(uv,n,'-x')
    
end

xlim([min(xx), max(xx)])
box on
xlabel('Reaction coordinate')
ylabel('Probability')
