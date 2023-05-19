function[Record_states, step] = explore_the_network(t,K_biased,bias,xx,kT,T,cutoff,nodes,state_start,state_end)
%% TJ note: probability is not normalized.

M_t = expm(K_biased.*t); % transition matrix.
M_t = M_t'; %% TJ note, M_t is it the transposed? or not.
Record_states = zeros(1, T);
Record_states(1)=state_start;
for i = 2:1:T
   
    P = M_t(Record_states(i-1),:);
    new_state = randsample(nodes,1,true,P); % choose the new state according to the current probability distribution.
 
    Record_states(i) = new_state; % record the new state.
    %x_rec(i) = xx(new_state);
    if new_state == state_end
        step=i
        break
    end
end

end