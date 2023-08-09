function K_biased = bias_K_1D(K, bias)
    N = size(K,1);
    kT = 0.5981; 
    K_biased = zeros(N);

    for i = 1:N-1
        u_ij = bias(i+1) - bias(i);
        K_biased(i,i+1) = K(i,i+1) * exp(-u_ij/2/kT);
        K_biased(i+1,i) = K(i+1,i) * exp(u_ij/2/kT);
        K_biased(i,i) = 0;
    end

    K_biased(N,N) = 0;

    % Normalizing the rate matrix.
    for i = 1:N
        f = sum(K_biased(i,:));
        K_biased(i,i) =  -f;
    end
end
