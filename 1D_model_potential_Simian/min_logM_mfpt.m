function [mfpt_mid] = min_logM_mfpt(v,wx,wM,cutoff,Ngauss,amp,kT,state_start,state_end)
% Computes Kemeny Gradient and Adam expression and compares the two
%   Inputs: 
%   K = Rate Matrix
%   Ngauss = Number of biasing Gaussians
t = 0.01; % time steps NEEDS to be a parameter later
N = size(wM,2);
bias=zeros(N,1);
% C_g(1:Ngauss) = v(1:Ngauss);
% std_g(1:Ngauss) = v(Ngauss+1:2*Ngauss);
% for i = 1:1:Ngauss
%     bias(1:N) = bias(1:N) + amp * exp(-(wx' - C_g(i)).^2/(2*std_g(i)^2));
% end
K0=logm(wM);
%K0=max(K0,0);
%K0=abs(K0);
for i = 1:N
    for j=1:N
        u_ij=bias(j)-bias(i);
        u_ij=min(cutoff,u_ij);
        u_ij=max(-cutoff,u_ij);
        K_biased(i,j)=K0(i,j)*exp(-u_ij/2/kT);
    end
    K_biased(i,i)=0;
end
%% Normalizing the rate matrix.
    for i = 1:N
        f=sum(K_biased(i,:));
        K_biased(i,i) =  -f;
    end
%% biased MFPTs
M_biased = expm(K_biased*t); % transition matrix.
[pi_biased, ~] = compute_free_energy(M_biased', kT);
Mmfpt=Markov_mfpt_calc(pi_biased',M_biased);
mfpt_mid = Mmfpt(state_start, state_end);
% kemeny=zeros(N,1);
% for i=1:N
%     for j=1:N
%             kemeny(i)=kemeny(i)+Mmfpt(i,j)*pi_biased(j);
%     end
% end
% min(kemeny)
% max(kemeny)
