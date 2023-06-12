%this helps perturbe the 2D K matrix.
%K is the unperturbed 2D K matrix, shape in (N*N, N*N).
%total_bias in shape (N,N).
%we go through each element of K and use equation k_biased = k*exp(e_diff/2kT) 
function [K_biased] = perturb_K_2d(K, total_bias, N)
    kT=0.5981;
    K_biased = zeros(N*N, N*N);
    
    for i = 1:N-1
        for j = 1:N
            index = i + N*(j-1); %flatten the 2D index.
            K_biased(index, index+1) = K(index, index+1)*exp((total_bias(i+1,j) - total_bias(i,j))/2/kT); 
            K_biased(index+1, index) = K(index+1, index)*exp((total_bias(i,j) - total_bias(i+1,j))/2/kT);
        end
    end

    for i = 1:N
        for j = 1:N-1
            index = i + N*(j-1); %flatten the 2D index.
            K_biased(index, index+N) = K(index, index+N)*exp((total_bias(i,j+1) - total_bias(i,j))/2/kT);
            K_biased(index+N, index) = K(index+N, index)*exp((total_bias(i,j) - total_bias(i,j+1))/2/kT);
        end
    end
    
    for i = 1:N*N
        K_biased(i,i) = -sum(K_biased(:, i));
    end
end