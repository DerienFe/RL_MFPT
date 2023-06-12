%TW on 31st May 2023
%this is the perturbation function, that initialize a set of random gaussians in 2d
%then apply it on the K matrix. using functions: perturb_K_2d.m and gaussian2d.m
%input: K - the unperturbed rate matrix.
%output: the optimal perturbed K matrix; opt bias; opt mfpt; opt gaussian params.

function [opt_K, opt_bias, opt_mfpt, opt_params] = random_find_min_mfpt(K, num_iterations, state_start, state_end, num_gaussians)
    kT = 0.5981;
    N = sqrt(size(K, 1));
    ts = 0.1
    %convert indexing.
    [from_i, from_j] = coord_to_index(state_start(1), state_start(2));
    [to_i, to_j] = coord_to_index(state_end(1), state_end(2));

    % Create grid for x and y values
    x = linspace(-3, 3, N); % Define the range and number of points for x-axis
    y = linspace(-3, 3, N); % Define the range and number of points for y-axis
    [X, Y] = meshgrid(x, y); % Create 2D grid for x and y values
    
    [peq,~] = compute_free_energy(K, kT);
    mfpts = mfpt_calc_2d(peq, K); % Initialize min MFPT with unperturbed 

    min_mfpt = mfpts(from_i, from_j, to_i, to_j);

    opt_K = K;
    opt_bias = zeros(size(X));
    opt_params = [];
    opt_mfpt = min_mfpt;
    
    for iteration = 1:num_iterations
        gaussian_centers = rand(num_gaussians, 2) * 6 - 3; % the x0, y0
        gaussian_widths = rand(num_gaussians, 2) * 0.5 + 0.5; % the sigma_x, sigma_y
        

        total_bias = zeros(size(X));
        for i = 1:num_gaussians
            total_bias = total_bias + gaussian2d(X, Y, gaussian_centers(i, 1), gaussian_centers(i, 2), gaussian_widths(i, 1), gaussian_widths(i, 2));
        end
        
        K_biased = perturb_K_2d(K, total_bias, N);
        [peq_biased, ~] = compute_free_energy(K_biased, kT); 
        %mfpts_biased = mfpt_calc_2d(peq_biased, K_biased);
        
        %switch to Adam's expression to lead random_find.
        M_t = expm(K_biased * ts); % transition matrix.
        Mmfpt=Markov_mfpt_calc(peq_biased', M_t');
    
        mfpt_biased = Mmfpt(from_i, from_j, to_i, to_j)*ts;
        
        % Update opt_K, opt_bias, opt_mfpt, and opt_params if a lower value is found
        if mfpt_biased < opt_mfpt
            opt_K = K_biased;
            opt_bias = total_bias;
            opt_mfpt = mfpt_biased;
            opt_params = [gaussian_centers, gaussian_widths];
            disp(['MFPT updated (Adam expr): ', num2str(mfpt_biased)]);
            %kemeny_check_matrix = kemeny_check(mfpts_biased, peq_biased); %perform a check every update.
        end
    end
    
    % Display the minimized MFPT
    disp(['Minimized MFPT: ', num2str(opt_mfpt)]);
end
