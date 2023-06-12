function[eig_indx] = several_umbrella_samp_exper(...
    Num_exper, ...
    num_windows, ...
    T, ...
    t, ...
    xx, ...
    eig_indx, ...
    n_states, ...
    K, ...
    kT, ...
    force_constant, ...
    pi, ...
    F, ...
    eigenvectors, ...
    eigenvalues_sorted, ...
    index,...
    cutoff, ...
    nodes)
%% Umbrella sampling parameters.

initial_agent_pos = find(xx == min(xx),1);
x_sorted = sort(xx);
x_eq = linspace(x_sorted(1), x_sorted(end), num_windows);

%% Initialzing vectors to store data

Record_states = zeros(num_windows, T, Num_exper);
Record_states(1,1,:) = initial_agent_pos;
x_rec = zeros(num_windows, T, Num_exper);
x_rec(1,1,:) = xx(Record_states(1,1,1));
pi_biased = zeros(num_windows, n_states, Num_exper);

%%

for j = 1:1:Num_exper

    for i = 1:1:num_windows


        [Record_states(i,:,j),  pi_biased(i,:,j), x_rec(i,:,j), ~] = explore_the_network( ...
            t, ...
            K, ...
            Record_states(i,:,j), ...
            x_eq(i), ...
            force_constant, ...
            xx', ...
            kT, ...
            x_rec(i,:,j), ...
            T, ...
            cutoff, ...
            nodes);

        if i < num_windows % here we assign the next configuration such that it is closest to the center of the next umbrella.

            Record_states(i+1,1,j) = Record_states(i,end,j);
            x_rec(i+1,1,j) = x_rec(i,end,j);

        end
        

    end

end

save(['data_eigenvector_', num2str(eig_indx), '_.mat'],'-v7.3')
disp([num2str(eig_indx), ' is finished'])

end



