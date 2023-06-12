function[Record_states, pi_biased, x_rec, M_t] = explore_the_network( ...
    t, ...
    K, ...
    Record_states, ...
    x_eq, ...
    k, ...
    xx, ...
    kT, ...
    x_rec, ...
    T, ...
    cutoff, ...
    nodes)
%%  Bias the transition rate matrix.

K_biased = zeros(size(K));
for i = 1:1:length(xx)
    
    x_i = xx(i);
    x_j = xx';
    
    u_ij = 0.5.*k.*( (x_j - x_eq).^2 - (x_i - x_eq).^2 );
    u_ij(u_ij>cutoff) = cutoff;
    u_ij(u_ij<-cutoff) = -cutoff;

    KK = K(i,:);

    KK = KK'.*exp(u_ij./(2*kT));
    K_biased(i,:) = KK';
    K_biased(i,i) = 0;
    
end

%% Normalizing the transition rate matrix.

for i = 1:1:size(K_biased,1)
    
    f=sum(K_biased(:,i));
    K_biased(i,i) =  -f;
    
end

%%
M_t = expm(K_biased.*t); % transition matrix.
M_t = M_t';

for i = 2:1:T
   
    P = M_t(Record_states(i-1),:);
    
    new_state = randsample(nodes,1,true,P); % choose the new state according to the current probability distribution.
    
    Record_states(i) = new_state; % record the new state.
    x_rec(i) = xx(new_state);
    
end

[pi_biased, ~, ~, ~, ~, ~] = compute_free_energy(K_biased, kT); % compute the based staionary distribution from K_biased.


end