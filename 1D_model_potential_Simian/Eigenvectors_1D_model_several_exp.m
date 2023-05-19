%%

clear all
clc

N=100;
n_states=N;
kT=0.5981;

%% Create a trasition rate matrix

Barrier_height_1 = 4;
Barrier_height_2 = 6;
K = create_K_1D(N, kT, Barrier_height_1, Barrier_height_2);

%% obtain free energy term.

[pi, F, eigenvectors, eigenvalues, eigenvalues_sorted, index] = compute_free_energy(K, kT);

%% Choose the projection coordinate.

eig_vecs = [1,2,3,4,5]; % first eigenvector is unbiased simulations.
xx = [];

for i = 1:1:numel(eig_vecs)

    xx{i} = eigenvectors(:,index(eig_vecs(i)),:);
%       xx{i} = (1:1:n_states)'; % Ordered projection_coordinates

end

%% Initialize

T = 10^3;
t = 100; % time steps.
num_windows = 90; % num windows.
Num_exper = 300;
nodes = linspace(1,n_states,n_states);
force_constant = 800*ones(1,numel(eig_vecs)); % force constant. 600 for xx = eigenvectors(:,index(2))', 1 for xx = y, 200 for networks 2, xx = eigenvectors(:,index(2))'
force_constant(1) = 0; % unbiased_simulations
cutoff = 20;

%%
parfor  i = 1:numel(eig_vecs)

    [eig_indx] = several_umbrella_samp_exper(Num_exper, ...
                                            num_windows, ...
                                            T, ...
                                            t, ... 
                                            xx{i}, ...
                                            eig_vecs(i), ...
                                            n_states, ...
                                            K, ...
                                            kT,...
                                            force_constant(i), ...
                                            pi, ...
                                            F, ...
                                            eigenvectors, ...
                                            eigenvalues_sorted, ...
                                            index,...
                                            cutoff, ...
                                            nodes);

end





