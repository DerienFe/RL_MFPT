function[steps,trajectory,not_reached] = propagate_N_steps(M_t,sim_time,nodes,state_start,state_end,trajectory,tot_time,not_reached)
%%
P = M_t(state_start,:);
for i = 1:sim_time
    new_state = randsample(nodes,1,true,P); % choose the new state according to the current probability distribution.
    traj(i) = new_state; % record the new state.
    if new_state == state_end
        steps=i;
        trajectory(tot_time+1:tot_time+steps)=traj;
        not_reached=false;
        break
    end
    P = M_t(traj(i),:);
end
if not_reached
    steps=sim_time;
    trajectory(tot_time+1:tot_time+steps)=traj;
end
end