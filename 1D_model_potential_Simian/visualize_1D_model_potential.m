
clear all
clc


N=100;
n_states=N;
kT=0.5981;
Barrier_height_1 = 8;
Barrier_height_2 = 12;

x=linspace(0,5*pi,N);
y1=Barrier_height_1*sin((x-pi));
y2=Barrier_height_2*sin((x-pi)/2);
A=10;
xtilt=0.5;
y=(xtilt*y1-y2);
y=xtilt*y1+(1-xtilt)*y2;

for i=1:N-1
    K(i,i+1)=A*exp((y(i+1)-y(i))/2/kT);
    K(i+1,i)=A*exp((y(i)-y(i+1))/2/kT);
end

for i=1:N
    K(i,i)=0;
    K(i,i)=-sum(K(:,i));
end

[prob_dist, F, eigenvectors, eigenvalues, eigenvalues_sorted, index] = compute_free_energy(K, kT);

figure
plot((1:n_states),F-min(F), LineWidth=2)
xlabel('nodes')
ylabel('probability')

figure
plot(1:1:n_states,prob_dist, LineWidth=2)
xlabel('nodes')
ylabel('Free energy')