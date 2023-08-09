function [NextObservation,Reward,IsDone,LoggedSignals] = myStepFunction(Action,LoggedSignals)
%
% This function applies the given action to the environment and evaluates
% the system dynamics for one simulation step.

    N=100;
    xx=(1:N);
    end_state = 89;
    kT=0.596;
    Ngauss=20;
    amp = 1.;
    ts = 0.01;
    sim_time = 1000;
    K = create_K_1D(N, kT);
    %size(Action)
    C_g(1:Ngauss) = Action(1,1:Ngauss);
    std_g(1:Ngauss) = Action(2,1:Ngauss);
    bias=zeros(N,1);
    for i = 1:1:Ngauss
        bias(1:N) = bias(1:N) + amp * exp(-(xx' - C_g(i)).^2/(2*std_g(i)^2));
    end
    plot((1:N),bias,'--','Linewidth',1.5)   
    for i = 1:N-1
        u_ij=bias(i+1)-bias(i);
        K_biased(i,i+1)=K(i,i+1)*exp(-u_ij/2/kT);
        K_biased(i+1,i)=K(i+1,i)*exp(u_ij/2/kT);
        K_biased(i,i)=0;
    end
    K_biased(N,N)=0;
    %% Normalizing the rate matrix.
    for i = 1:N
        f=sum(K_biased(i,:));
        K_biased(i,i) =  -f;
    end
    %% biased MFPTs
    M = expm(K_biased*ts); % transition matrix
    [p,F,~]=compute_free_energy(M', kT);
    plot((1:N),F,'-','Linewidth',1.5)  
    trajectory=[];
    % Propagate the system.
    [steps,trajectory,IsDone] = propagate_N_steps(M,sim_time,xx,LoggedSignals.State,end_state,trajectory);
    NextObservation = trajectory(steps);
    %get reward
    %Reward = NextObservation - end_state;
    farest = max(trajectory);
    Reward = (farest - end_state);
    farest

    %avg_pos = mean(trajectory);
    %Reward = (avg_pos - end_state);
    %avg_pos

    
    %second part of reward, time penalty.
    if ~IsDone
        Reward = Reward-10;
    else
        Reward = Reward + 100;
    

    %third part
    % if placed bias is not within explored region, then penalty.
    % because this means our action has no impact.
    traj_min = min(trajectory);
    traj_max = max(trajectory);
    for i = 1:1:Ngauss
        if C_g(i) < traj_min && C_g(i) > traj_max
            Reward = Reward - 10;
        end
    end

end