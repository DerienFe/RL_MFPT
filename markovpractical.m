% Edina Rosta
% edina.rosta@kcl.ac.uk
%%
clear all 
%close all
% N = number of states
% K(i,j) rate constant for the i --> j process
% rand is a subroutine that generates a uniformly
% distributed random number between (0,1)
kB=0.0019872041; % Boltzmann constant (kcal/mol)
temp=298; % Temperature
KbT=kB*temp;
N=10;
K=create_K_1D(N,KbT);
%% 

%calculate equilibrium from spectral decomposition
[eigvec,eigval]=eig(K); % diagonalize K, eigvec stores the eigenvectors, eigval the eigenvalues
[dsorted,index]=sort(diag(eigval),'descend'); % sort the eigenvalues. dsorted stores the eigenvalues, index the corresponding indices
% sorted eigenvalues:
ind=index(1);
eq=eigvec(:,ind)/sum(eigvec(:,ind)) % equilibrium probability corresponds to 0 eigenvalue. Based on the equilibrium probability we can also obtain the energy.
figure
hold on
x=linspace(0,1,10)
for i=1:N
plot(x,linspace(dsorted(i),dsorted(i),10),'LineWidth',3)
end
ylabel('Eigenvalue','FontSize',18)
splitting=-(dsorted(2)-dsorted(3))/dsorted(2); % calculate the splitting based on a two state approach
title(['Two-state splitting of the eigenspectrum =',num2str(splitting)]);
%% 
figure
hold on
xlabel('# State','FontSize',18)                   
ylabel(['Equilibrium probability'],'FontSize',18)
bar(eq,'r')

energy=kB*temp*(-log(eq));% calculate the energy
energy=energy-min(energy); %
% Now we plot the energies...
figure
hold on
xlabel('# State','FontSize',18)                   
ylabel(['\DeltaG (kcal/mol)'],'FontSize',18)
bar(energy,'r')
%plot(energy,'b-o','MarkerSize',10)
hold off
%% 

% splitting and eigvec
[eigvec,eigval]=eig(K'); % diagonalize K, eigvec stores the right eigenvectors
dsorted
slowest_relrate=-dsorted(2) 
slow_vec=eigvec(:,index(2));
[dsorted,index]=sort(diag(eigval),'descend'); % sort the eigenvalues. 
figure
hold on
bar(slow_vec)
xlabel('# State','FontSize',18)
ylabel('Second eigenvector','FontSize',18)
hold off
%% initialization of the trajectories
s_traj(1)=1;
Nstep=100000;
stepsize=0.1;
M=expm(K'*stepsize);
% generating trajectories
for i = 2:1:Nstep
    P = M(s_traj(i-1),:);
    new_state = randsample((1:N),1,true,P); % choose the new state according to the current probability distribution.
    s_traj(i) = new_state; % record the new state.
end
mfpt_c = mfpt_dat(s_traj,N,stepsize)
mfpt_M = Markov_mfpt_calc(eq,M,N,stepsize)
mfpt_K = mfpt_calc(eq,K,N)

% Construct Markov chain
lagtime=1;
qspace=(1:N+1);
ncount(1:N+1)=histc(s_traj(1:end-lagtime),qspace);

MM=zeros(N,N);
for i=1+lagtime:Nstep
    MM(s_traj(i-lagtime),s_traj(i))=MM(s_traj(i-lagtime),s_traj(i))+1;
end

msum=sum(MM');
for i=1:N
    if msum(i) > 0
        MM(i,:)=MM(i,:)/msum(i);
    else
        MM(i,:)=0;
    end
end
MM'
%%
% this is equivalent with the matrix exponential of the rate matrix:
%% 

[eigvecM,eigvalM]=eig(MM'); % diagonalize K, eigvec stores the eigenvectors, eigval the eigenvalues
[dsortedM,indexM]=sort(diag(eigvalM),'descend'); % sort the eigenvalues. dsortes stores the eigenvalues, index the corresponding indices
slowest_relrateM=-log(dsortedM(2))/0.05 % compare to slowest_relrate=-dsorted(2)
slowest_relrate
ind=indexM(1);
dsortedM

% the equilibrium eigenvector from the Markov chain:
eqM=eigvecM(:,ind)/sum(eigvecM(:,ind))

% Compare analytical and measured p_eq
figure
hold on
xlabel('# State','FontSize',18)
ylabel(['Equilibrium probability'],'FontSize',18)
bar(eq,'r')
bar(eqM,'b', 'BarWidth',0.4)
legend('exact','Markov chain')

figure
hold on
x=linspace(0,1,10)
for i=1:N
plot(x,linspace(dsortedM(i),dsortedM(i),10),'LineWidth',3)
end
ylabel('Eigenvalue','FontSize',18)


