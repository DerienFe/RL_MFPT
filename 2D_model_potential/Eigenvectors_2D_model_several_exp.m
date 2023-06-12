%%

clear all
clc

N = 13;
n_states = N*N;
kT=0.5981;
ts = 0.1;

state_start = [-2, -2];
state_end = [2, 2];
%% Create a trasition rate matrix
barrier_height = 0.7; % default 0.7.
[K,Z] = create_K_2D(N, kT, barrier_height);

%% obtain free energy term.

[pi, F, eigenvectors, eigenvalues, eigenvalues_sorted, index] = compute_free_energy(K, kT);

%% initialize the 2d gaussians

num_gaussians = 20; 
gaussian_centers = rand(num_gaussians, 2) * 6 - 3; % the x0, y0
gaussian_widths = rand(num_gaussians, 2) * 0.5 + 0.5; % the sigma_x, sigma_y

x = linspace(-3, 3, N); % Define the range and number of points for x-axis
y = linspace(-3, 3, N); % Define the range and number of points for y-axis
[X, Y] = meshgrid(x, y); % Create 2D grid for x and y values

% Use the gaussian_2d function to create the perturbation
total_bias = zeros(size(X));
for i = 1:num_gaussians
    total_bias = total_bias + gaussian2d(X, Y, gaussian_centers(i, 1), gaussian_centers(i, 2), gaussian_widths(i, 1), gaussian_widths(i, 2));
end

% Plot the total bias
%contourf(X, Y, total_bias');
%title('total bias after 10 random gaussians');
%colorbar;

K_biased = perturb_K_2d(K, total_bias, N);
[peq_biased, ~] = compute_free_energy(K, kT);
mfpt_biased = mfpt_calc_2d(peq_biased, K_biased);

[from_i, from_j] = coord_to_index(state_start(1), state_start(2));
[to_i, to_j] = coord_to_index(state_end(1), state_end(2));

disp(['MFPT perturbed: ', num2str(mfpt_biased(from_i, from_j, to_i, to_j))]);
kemeny_check_matrix = kemeny_check(mfpt_biased, peq_biased);

%% Try the Markov mfpt calc.
M_t = expm(K_biased*ts); % transition matrix. in shape (N*N, N*N)
Mmfpt=Markov_mfpt_calc(peq_biased', M_t');
%kemeny_check_matrix = kemeny_check(Mmfpt, peq_biased);
%Mmfpt_poi = Mmfpt(state_start, state_end)*ts;
%Mmfpt = Mmfpt * ts
disp(['MFPT perturbed using Markov mfpt calc: ', num2str(Mmfpt(from_i, from_j, to_i, to_j) * ts)]);

%% random try place the gaussian 1000 times.
num_iterations = 1000;
[opt_K, opt_bias, opt_mfpt, opt_params] = random_find_min_mfpt(K, num_iterations, state_start, state_end, num_gaussians);

% Plot the total bias
figure
contourf(X, Y, opt_bias','LevelStep',0.5,'Fill','on');
title('total bias after optimized gaussians');
colorbar;

figure
contourf(X, Y, (opt_bias+Z)','LevelStep',0.5,'Fill','on');
title('total potential after optimal gaussians');
colorbar;

%% here we apply the opt_params, wrap up the function into min_mfpt_helper.m
% then use fminsearch to find the best params.
%comment out if there's opt_params_fmin.mat data. just load it.
params_0 = opt_params;
opt_params_fmin = fminsearch(@(opt_params) min_mfpt_helper(opt_params, K, num_gaussians, ts, state_start, state_end), params_0, optimset('MaxFunEvals',1e7));

%apply the opt_params_fmin to the FES and visualize.
gaussian_centers = opt_params_fmin(1:end, 1:2);
gaussian_widths = opt_params_fmin(1:end, 3:end);

total_bias = zeros(N);
for i = 1:num_gaussians
    total_bias = total_bias + gaussian2d(X, Y, gaussian_centers(i, 1), gaussian_centers(i, 2), gaussian_widths(i, 1), gaussian_widths(i, 2));
end

%plot the fmin optimized total bias.
figure
contourf(X, Y, total_bias','LevelStep',0.5,'Fill','on');
title('total bias after fmin optimized gaussians');
colorbar;

figure
contourf(X, Y, (total_bias+Z)','LevelStep',0.5,'Fill','on');
title('total potential after fmin optimized gaussians');
colorbar;

%save the global status as .mat data here for book keeping.
save('opt_params_fmin.mat', 'opt_params_fmin');

%% here we try global optimization. Initialize the gaussian evenly across the grid.
% the optimizer is having a hard time. consider start with best random
% search (above), or set the searching barrier around (-2, -2).
% first we use multistart, which evaluate each starting point equally.
% then we use global minimum search, which pre-evaluate then apply the
% search.

%note TW: random x0 fails to converge, try opt_params_fmin as x0 in GS/MS.

%initialize the gaussian centers and widths.
gaussian_centers_0 = zeros(num_gaussians, 2);
gaussian_widths_0 = zeros(num_gaussians, 2);

%initialize the gaussian centers and widths evenly across the grid.
for i = 1:num_gaussians
    gaussian_centers_0(i, 1) = -3 + (i-1) * 6 / num_gaussians;
    gaussian_centers_0(i, 2) = -3 + (i-1) * 6 / num_gaussians;
    gaussian_widths_0(i, 1) = 0.5;
    gaussian_widths_0(i, 2) = 0.5;
end

%here we define the objective function for the global search. the mfpt.
%we use the min_mfpt_helper function to wrap up the function.

%replace this line in createOptimProblem() to use random x0 to start search:
% 'x0', [gaussian_centers_0, gaussian_widths_0],...
options = optimoptions(@fmincon, 'MaxIter', 1e7, 'FunctionTolerance', 1e-3, 'StepTolerance', 1e-3, 'Algorithm', 'sqp');
problem = createOptimProblem('fmincon','objective',...
                            @(opt_params) min_mfpt_helper(opt_params, K, num_gaussians, ts, state_start, state_end),...
                            'x0', opt_params_fmin,...
                            'lb', [-3*ones(num_gaussians, 2), 0.5*ones(num_gaussians, 2)],...
                            'ub', [3*ones(num_gaussians, 2), 1.5*ones(num_gaussians, 2)],...
                            'options', options);

%here is the global-search algorithm.
gs = GlobalSearch('Display','iter',...
                  'MaxTime', 1800,...
                  'FunctionTolerance', 1e-3,...
                  'XTolerance', 1e-3);
[opt_params_gs, opt_mfpt_gs] = run(gs, problem);
save('opt_params_gs.mat', 'opt_params_gs');


% plot global opt result.
gaussian_centers_gs = opt_params_gs(1:end, 1:2);
gaussian_widths_gs = opt_params_gs(1:end, 3:end);

total_bias_gs = zeros(N);
for i = 1:num_gaussians
    total_bias_gs = total_bias + gaussian2d(X, Y, gaussian_centers_gs(i, 1), gaussian_centers_gs(i, 2), gaussian_widths_gs(i, 1), gaussian_widths_gs(i, 2));
end

figure
contourf(X, Y, total_bias_gs','LevelStep',0.5,'Fill','on');
title('total bias after global search optimized gaussians');
colorbar;

figure
contourf(X, Y, (total_bias_gs+Z)','LevelStep',0.5,'Fill','on');
title('total potential after global search optimized gaussians');
colorbar;


% here is the multi-search algorithm.
ms = MultiStart(gs); %use gs parameters set above.
num_starting_points = 10;
[opt_params_ms, opt_mfpt_ms] = run(ms, problem, num_starting_points);

save('opt_params_ms.mat', 'opt_params_ms');

% plot multi-search opt result.
gaussian_centers_ms = opt_params_ms(1:end, 1:2);
gaussian_widths_ms = opt_params_ms(1:end, 3:end);

total_bias_ms = zeros(N);
for i = 1:num_gaussians
    total_bias_ms = total_bias + gaussian2d(X, Y, gaussian_centers_ms(i, 1), gaussian_centers_ms(i, 2), gaussian_widths_ms(i, 1), gaussian_widths_ms(i, 2));
end

figure
contourf(X, Y, total_bias_ms','LevelStep',0.5,'Fill','on');
title('total bias after multi-start search optimized gaussians');
colorbar;

figure
contourf(X, Y, (total_bias_ms+Z)','LevelStep',0.5,'Fill','on');
title('total potential after multi-start search optimized gaussians');
colorbar;






