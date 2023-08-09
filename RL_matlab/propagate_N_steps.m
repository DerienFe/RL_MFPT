function[steps,trajectory,IsDone] = propagate_N_steps(M_t,sim_time,nodes,state_start,state_end,trajectory)
%%
IsDone=false;
P = M_t(state_start,:);
for i = 1:sim_time
    new_state = randsample(nodes,1,true,P); % choose the new state according to the current probability distribution.
    traj(i) = new_state; % record the new state.
    if new_state == state_end
        steps=i;
        trajectory(tot_time+1:tot_time+steps)=traj;
        IsDone=true;
        break
    end
    P = M_t(traj(i),:);
end
if ~IsDone
    steps=sim_time;
    trajectory(1:steps)=traj;
end
end