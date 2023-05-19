function [MFPT] = MFPT(traj,number_state)

counts_step = zeros(number_state,number_state);
counts_number = zeros(number_state,number_state);
for i = 1:size(traj,2)
        for j = (i+1):size(traj,2)
            if traj(i) - traj(j) ~= 0
                counts_step(traj(i),traj(j)) = counts_step(traj(i),traj(j)) + j - i;
                counts_number(traj(i),traj(j)) = counts_number(traj(i),traj(j)) + 1;
            end
        end
end
for i = 1:number_state
    for j = 1:number_state
        MFPT(i,j) = counts_step(i,j) / counts_number(i,j);
    end
end
end

