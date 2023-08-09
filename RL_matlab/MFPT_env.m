%here we create a RL environment with matlab RL toolkit.
%getObservationInfo	Return information about the environment observations
%getActionInfo	Return information about the environment actions
%sim	Simulate the environment with an agent
%validateEnvironment	Validate the environment by calling the reset function and simulating the environment for one time step using step
%reset	Initialize the environment state and clean up any visualization
%step	Apply an action, simulate the environment for one step, and output the observations and rewards; also, set a flag indicating whether the episode is complete
%Constructor function	A function with the same name as the class that creates an instance of the class

%this environment do 20 Gaussian at the same time.
%so we don't need to propagate the K.

classdef MFPT_env < rl.env.MATLABEnvironment
    %initialize properties
    properties
        N = 100;
        kT = 0.5981;
        state_start = 8;
        state_end = 89;
        ts = 0.01;
        sim_incr = 1000;
        num_actions = 20;

        K;
        peq;
        mfpts;
        mfpt;
    end

    %define state space.
    properties
        state;
    end

    %define action space.
    %where we put the gaussians:
    %  the position can be chose from (1,N)
    %  the amp is fixed to 1, the width can chose from (0.5, 1, 1.5, 2) 
    properties(Access = protected)
        IsDone = false;
        C_g;
        std_g;
        
        action_space;
    end
    
    methods
        
        %this section initialize the environment object "this".
        function this = MFPT_env()
            
            % Initialize Observation settings
            ObservationInfo = rlFiniteSetSpec([1:100]);
            ObservationInfo.Name = 'position';
            ObservationInfo.Description = 'x-axis position of the particle';

            %initialize the action space.
            % note we put 20 gaussian at one time. each gaussian has position and width.
            % we put upper/lower bound to the position and width. [1, N] and [0.5, 2]
            lowerLimits(1,1:20)=0;
            lowerLimits(2,1:20)=0.1;
            upperLimits(1,1:20)=100;
            upperLimits(2,1:20)=5;

            ActionInfo = rlNumericSpec([2,20],'LowerLimit', lowerLimits, 'UpperLimit',upperLimits);
            ActionInfo.Name = 'gaussian position and width';

            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);

            %initialize the complex env properties listed above.
            this.state = zeros(this.N, 1); %one-hot style position marker
            this.C_g = 1:this.N;
            this.std_g = [0.5, 1, 1.5, 2];
            this.action_space = zeros(length(this.C_g), length(this.std_g), this.num_actions);
            
            this.K = create_K_1D(this.N, this.kT);
            [this.peq,~] = compute_free_energy(this.K', this.kT);
            this.mfpts = mfpt_calc(this.peq, this.K);
            this.mfpt = this.mfpts(this.state_start, this.state_end);

        end
        function [InitialObservation, LoggedSignal] = myResetFunction()
% Reset function to place custom cart-pole environment into a random
% initial state.

% Theta (randomize)
T0 = 2 * 0.05 * rand() - 0.05;
% Thetadot
Td0 = 0;
% X
X0 = 0;
% Xdot
Xd0 = 0;

% Return initial environment state variables as logged signals.
LoggedSignal.State = [X0;Xd0;T0;Td0];
InitialObservation = LoggedSignal.State;

end
        %this section define the reset functionality under the environment object "this".
        function [InitialObservation,LoggedSignals] = reset(this)
            % reset to the start state
            InitialObservation = this.state;
            InitialObservation(:) = 0;
            InitialObservation(this.state_start) = 1; %set the starting state is 1.
            LoggedSignals = [];
            this.IsDone = false;

            notifyEnvUpdated(this);
        end

        %this section define the step functionality under the environment object "this".
        function [NextObservation,Reward,IsDone, LoggedSignals] = step(this, Action)
            LoggedSignals = [];

            C_gs = Action(:,1); %unpack gaussian parameters C_gs and std_gs has 20 elements
            std_gs = Action(:, 2);

            total_bias = zeros(this.N,1);
            for i = 1:length(C_gs)
                total_bias = total_bias + gaussian_bias(C_gs(i), std_gs(i), this.N);
            end

            % Cache to avoid recomputation?

            %now we have the total bias, we calculate the biased K matrix. and M matrix.
            % then we use the M matrix, propagate system 1000 steps.
            % update the new state as the propagated state.

            K_biased = bias_K_1D(this.K, total_bias);
            M = expm(K_biased * this.ts);
            M = M./sum(M,2); %normalize the transition matrix.
            
            % Get current state index
            currentStateIdx = find(this.state == 1);
        
            % Propagate the system.
            NextObservation = this.state;
            for i = 1:this.sim_incr
                p = M(currentStateIdx, :);
                NextObservation(:) = 0;  % Reset the state vector
                nextStateIdx = randsample(this.N, 1, true, p);
                NextObservation(nextStateIdx) = 1;  % Set the next state in the state vector
                currentStateIdx = nextStateIdx;
            end
        
            % Update system states
            this.state = NextObservation;

            %check if meets terminal condition
            %if the state is the end state, then the episode is done.
            if find(NextObservation == 1) == this.state_end
                this.IsDone = true;
            end

            %get reward
            Reward = getReward(this);
            IsDone = this.IsDone;
            notifyEnvUpdated(this);
        end

        %this section define the getReward functionality under the environment object "this".
        function [Reward] = getReward(this)
            %Reward = this.state - this.state_end; %the reward is the distance to the end state.
            if ~this.IsDone
                Reward = 0; %at the end, reward is 0.
            else
                distance = find(NextObservation == 1) - this.state_end;
                Reward = distance; %the reward is the distance between current pos to the end state pos.
        end

        end
    end
end

%validateEnvironment(MFPT_env);