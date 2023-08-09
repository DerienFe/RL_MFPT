function [InitialObservation, LoggedSignal] = myResetFunction()
% Reset function to place the system in starting state
% Return initial environment state variables as logged signals.
%close all
state_start = 8;
LoggedSignal.State = state_start;
InitialObservation = LoggedSignal.State;
IsDone=false;
close all
figure
    N=100;
    xx=(1:N);
    end_state = 89;
    kT=0.596;
    Ngauss=20;
    amp = 1.;
    ts = 0.01;
    sim_time = 1000;
    K = create_K_1D(N, kT);
    [p,F,~]=compute_free_energy(K', kT);
    figure
plot(xx,F,'b', 'LineWidth', 2)
hold on

end

