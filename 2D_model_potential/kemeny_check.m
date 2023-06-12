%the 2D version of kemeny check. if the max/min of the matrix is the same,
%then it passed the check.
function kemeny = kemeny_check(mfpt, peq)
    % N is the number of states.
    % mfpt is the mean first passage time matrix in shape (N, N, N, N)
    % peq is the stationary distribution
    N = size(mfpt, 1);
    kemeny = zeros(N, N);
    peq_2d = reshape(peq, [N, N]);
    for i = 1:N
        for j = 1:N
            for k = 1:N
                for l = 1:N
                    kemeny(i, j) = kemeny(i, j) + mfpt(i, j, k, l) * peq_2d(k, l);
                end
            end
        end
    end

    % Print the min/max of the Kemeny constant
    disp(['The min/max of the Kemeny constant is: ', num2str(min(kemeny(:))), ' ', num2str(max(kemeny(:)))]);
end
