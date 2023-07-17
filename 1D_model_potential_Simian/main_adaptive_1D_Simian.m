%%
clear all
clc

N=100;
kT=0.596;

%% Create a trasition rate matrix

K = create_K_1D(N, kT);

%% obtain free energy term.

[eq, F, eigenvectors, eigenvalues, eigenvalues_sorted, index] = compute_free_energy(K',kT);
F=F-min(F);
[mfpts] = mfpt_calc(eq,K);
state_start=1;
state_end = 89;
mfpt0 = mfpts(state_start, state_end)
%% Kemeny constant check
kemeny=zeros(N,1);
for i=1:N
    for j=1:N
            kemeny(i)=kemeny(i)+mfpts(i,j)*eq(j);
    end
end
min(kemeny)
max(kemeny)
Kemeny_eig=-sum(1./eigenvalues_sorted(2:N))

%% cutoff for numerical exp issues
cutoff = 20;
xx = linspace(1,N,N);

% bias size and type
Ngauss=20;
amp = 1.;
t = 0.01; % time steps

get_random_guess; % initial guess from 100 random bias attempts
%%%
% a=v;
% LB = ones(1, Ngauss); % lower bound
% UB = repmat(N, 1, Ngauss); % upper bound

% for is=state_start:1:state_end
    % v0=a;
    
    % [a,b,c]= fmincon(@(v) min_mfpt(v, K, Ngauss, amp, kT, t, is, state_end), v0,[],[],[],[],LB,UB);
    % a
    % pos_bias{is}=a;
% end
% save pos_bias_m.mat pos_bias
load pos_bias_m.mat
%%
%%%% Initialize adaptive sims
Nsim=5;
initial_agent_pos=state_start;
T = 10^7;
nodes = linspace(1,N,N);
Record_states = zeros(Nsim, T);
Record_states(1,1) = initial_agent_pos;
sim_incr=1000;
for isim=1:Nsim
    Record_states(isim,1) = initial_agent_pos;
    current_pos=initial_agent_pos;
    not_reached=true;
    tot_time=0;
    trajectory=zeros(1, T);
     while not_reached
          v=pos_bias{current_pos};
          get_biased_M; %using v as the bias, and t as time steps
          % propagate the biased simulations for sim_incr steps
          [steps,trajectory,not_reached] = propagate_N_steps(M_t,sim_incr,nodes,current_pos,state_end,trajectory,tot_time,not_reached);
          tot_time=tot_time+steps;
          current_pos=trajectory(tot_time);
     end
    
     RR = nonzeros(trajectory');
     segmentIndices = 1:sim_incr:length(RR);

     figure
     hold on
     for i = 1:length(segmentIndices)-1
         
         startIndex = segmentIndices(i);
         endIndex = segmentIndices(i+1);
         x = startIndex:endIndex;
         color = rand(1, 3);
         
         plot(RR(startIndex:endIndex), x, 'Color', color, 'LineWidth', 2);
     end
         plot(RR(endIndex+1:length(RR)), endIndex+1:length(RR),'Color', color, 'LineWidth', 2);
     hold off
     
     xlabel('state')
     ylabel('timestep')
     ylim([0, N])
     title('Traceplot of Nodes Visited - Adaptive')
     mfpt_adaptive(isim)=tot_time
end

average_mfpt_adaptive=mean(mfpt_adaptive)*t
error_mfpt_stochastic=std(mfpt_adaptive)*t

%% using a single bias across the full sims
figure
ax1 = gca;
plot(ax1, xx, F, 'b', 'LineWidth', 2)
hold(ax1, 'on')
colorMap = jet(state_end - state_start + 1); % Use the jet colormap

for i = state_start:state_end
    v = pos_bias{i};
    C_g(1:Ngauss) = v(1:Ngauss);
    std_g(1:Ngauss) = v(Ngauss+1:2*Ngauss);
    bias = zeros(N, 1);
    for j = 1:Ngauss
        bias(1:N) = bias(1:N) + amp * exp(-(xx' - C_g(j)).^2 / (2 * std_g(j)^2));
    end
    % plot(xx, bias, 'r', 'LineWidth', 2)
    % hold on
    colorIndex = i - state_start + 1;
    color = colorMap(colorIndex, :);
    plot(ax1, xx, bias + F' - min(bias_opt + F'), '--', 'Color', color, 'LineWidth', 2);
    hold(ax1, 'on');
end

hold(ax1, 'off');
xlabel(ax1, 'state');
ylabel(ax1, 'FES');

% Create legend
legendStrings = cell(1, state_end - state_start + 1);
for i = state_start:state_end
    legendStrings{i - state_start + 1} = sprintf('State %d', i);
end
legend(ax1, legendStrings, 'Location', 'eastoutside');
hold off;

%%%
%% generating biased stochastic trajectory
for i = 1:N-1
    u_ij=bias_opt(i+1)-bias_opt(i);
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
[pi_biased, ~] = compute_free_energy(K_biased', kT);
plot(xx,-log(pi_biased)*kT-min(-log(pi_biased)*kT), 'c--', 'LineWidth', 2)

T = 10^7;
t = .01; % time lag
Nsim=10;
step_list = zeros(Nsim);
for i = 1:1:Nsim
    [Record_states(i,:),  step_list(i)] = explore_the_network(t,K_biased,bias_opt,xx,kT,T,cutoff,nodes,state_start,state_end);
    
    % Traceplot of nodes visited.
    RR = nonzeros(Record_states(i,:)');
    figure
    plot(RR(:),length(step))
    xlabel('time')
    ylabel('nodes')
    ylim([0, N])
    title('traceplot of nodes visited')
end
average_mfpt_stochastic=mean(step)*t
error_mfpt_stochastic=std(step)*t
mmm=mfpt_calc(pi_biased,K_biased');
mfpt_from_rates=mmm(state_start, state_end)
M_t = expm(K_biased*t); % transition matrix.
Mmfpt=Markov_mfpt_calc(pi_biased',M_t);
mfpt_from_Markov=Mmfpt(state_start, state_end)*t
stoppoint

%% Figure for comparison
data = [mfpt0, mfpt_from_Markov, average_mfpt_stochastic, average_mfpt_adaptive];

figure;
bar(data);

title('Comparison for different biasing');
xlabel('biasing');
ylabel('mfpt');

xticks(1:numel(data));
xticklabels({'Unbiasing', 'Markov', 'Stochastic', 'Adaptive'});
xtickangle(45);

ylim([0, max(data)+1]);

