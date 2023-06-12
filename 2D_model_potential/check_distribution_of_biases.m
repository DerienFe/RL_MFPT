%
clear all
clc

%% 

force_constant = 350; % force constant
N = 13;
n_states = N*N;
kT=0.5981;

%% Create a trasition rate matrix

barrier_height = 0.7; % default 0.7.
K = create_K_2D(N, kT, barrier_height);

%% obtain free energy term.

[~, ~, eigenvectors, ~, ~, index] = compute_free_energy(K, kT);
lambda = 2;
xx = eigenvectors(:,index(lambda));

%% Initialize

num_windows = 10;
nodes = linspace(1,n_states,n_states);
x_sorted = sort(xx);
x_eq = linspace(x_sorted(1), x_sorted(end), num_windows);
cutoff =30;
%%  Bias the transition rate matrix.

figure
hold on
for j = 1:1:numel(x_eq)

    K_biased = [];
    for i = 1:1:length(xx)

        x_i = xx(i);
        x_j = xx';

        u_ij = 0.5.*force_constant.*( (x_j - x_eq(j)).^2 - (x_i - x_eq(j)).^2 );
        u_ij(u_ij>cutoff) = cutoff;
        u_ij(u_ij<-cutoff) = -cutoff;
        KK = exp(u_ij./(2*kT));

        K_biased(i,:) = KK';
        K_biased(i,i) = 0;

    end

    for i = 1:1:size(K_biased,1)

        f=sum(K_biased(:,i));
        K_biased(i,i) =  -f;

    end

    [pi_biased, F_biased, eigenvectors_biased, eigenvalues_biased, eigenvalues_sorted_biased, index_biased] = compute_free_energy(K_biased, kT);
    scatter(xx,pi_biased,'fill')


end

box on
xlabel(['reaction coordinate (\psi_', num2str(lambda) ')'])
ylabel('distirbution of biases')
xlim([min(xx), max(xx)])
title(['Force constant ' num2str(force_constant)])
ylim([0,0.05])