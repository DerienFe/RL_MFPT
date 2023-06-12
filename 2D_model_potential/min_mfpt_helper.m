function [mfpt_biased] = min_mfpt_helper(opt_params, K, num_gaussians, ts, state_start, state_end)

    %opt_params is a vector of the 2d gaussians (optimal in random search, to be optimized in fminsearch)
    %K is the rate matrix, unperturbed.
    %num_gaussians is the number of 2d gaussians.
    %ts is the time step.
    %state_start is the starting state.
    %state_end is the ending state.

    %initialize parameters
    N = sqrt(size(K,1)); %note that K is in shape (N*N, N*N)
    kT = 0.5981;

    %convert indexing.
    [from_i, from_j] = coord_to_index(state_start(1), state_start(2));
    [to_i, to_j] = coord_to_index(state_end(1), state_end(2));

    %compute total bias
    gaussian_centers = opt_params(1:end, 1:2);
    gaussian_widths = opt_params(1:end, 3:end);

    % Create grid for x and y values
    x = linspace(-3, 3, N); % Define the range and number of points for x-axis
    y = linspace(-3, 3, N); % Define the range and number of points for y-axis
    [X, Y] = meshgrid(x, y); % Create 2D grid for x and y values
    
    total_bias = zeros(N);
    for i = 1:num_gaussians
        total_bias = total_bias + gaussian2d(X, Y, gaussian_centers(i, 1), gaussian_centers(i, 2), gaussian_widths(i, 1), gaussian_widths(i, 2));
    end
    
    %perturb K matrix and get mfpt.
    K_biased_opt = perturb_K_2d(K, total_bias, N);
    [peq_biased_opt, ~] = compute_free_energy(K_biased_opt, kT); 
    %mfpts_biased_opt = mfpt_calc_2d(peq_biased_opt, K_biased_opt);
    %mfpt_biased = mfpts_biased_opt(from_i, from_j, to_i, to_j); %here is the mfpt from rate matrix.

    %here we use Adam's expression. using M transition matrix. Markov_mfpt_calc.m
    M_t = expm(K_biased_opt * ts); % transition matrix.
    Mmfpt=Markov_mfpt_calc(peq_biased_opt',M_t');
    mfpt_biased = Mmfpt(from_i, from_j, to_i, to_j)*ts;
end
