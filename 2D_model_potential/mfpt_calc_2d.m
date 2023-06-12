function mfpt = mfpt_calc_2d(peq, K)
    % peq is the probability distribution at equilibrium.
    % K is the transition matrix.
    % N is the number of states.

    % Here we output the mfpt in shape (N*N, N*N).
    % Each element is the mfpt from (i, j) to (k, l).

    N = sqrt(size(K, 1)); % Total states is N*N for a 2D grid
    onevec = ones(N*N, 1);
    peq_flat = peq(:);
    Qinv = inv(peq_flat.' * onevec - K.'); % Qinv is the inverse of the matrix Q 

    mfpt = zeros(N*N, N*N);
    for l = 1:N
        for k = 1:N
            for j = 1:N
                for i = 1:N
                    % Convert 2D indices to 1D index
                    idx_from = (i - 1) * N + j;
                    idx_to = (k - 1) * N + l;
                    
                    mfpt(idx_from, idx_to) = 1 / peq_flat(idx_to) * (Qinv(idx_to, idx_to) - Qinv(idx_from, idx_to));
                end
            end
        end
    end

    % Reshape mfpt into 2D grid shape
    % so that mfpt(i,j,k,l) is the MFPT from state (i,j) to (k,l)
    mfpt = reshape(mfpt, [N, N, N, N]); 
end
