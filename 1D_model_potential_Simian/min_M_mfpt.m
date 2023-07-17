function [mfpt_mid] = min_M_mfpt(v,wx,wM,cutoff,Ngauss,amp,kT,state_start,state_end)
% Computes Kemeny Gradient and Adam expression and compares the two
%   Inputs: 
%   K = Rate Matrix
%   Ngauss = Number of biasing Gaussians
N = size(wM,2);
bias=zeros(N,1);
C_g(1:Ngauss) = v(1:Ngauss);
std_g(1:Ngauss) = v(Ngauss+1:2*Ngauss);
for i = 1:1:Ngauss
    bias(1:N) = bias(1:N) + amp * exp(-(wx' - C_g(i)).^2/(2*std_g(i)^2));
end
for i = 1:N
    for j=1:N
        u_ij=bias(j)-bias(i);
        u_ij=min(cutoff,u_ij);
        u_ij=max(-cutoff,u_ij);
        M_biased(i,j)=wM(i,j)*exp(-u_ij/2/kT);
    end
    M_biased(i,i)=wM(i,i);
end
%% Normalizing the rate matrix.
for i = 1:N
    s=sum(M_biased(i,:));
    if s ~= 0
        M_biased(i,:) = M_biased(i,:)/s;
    end
end
%% biased MFPTs
[pi_biased, ~] = compute_free_energy(M_biased', kT);
pi_biased;
Mmfpt=Markov_mfpt_calc(pi_biased',M_biased);
%Mmfpt=jjhunter(M_biased);
mfpt_mid = Mmfpt(state_start, state_end)
% kemeny=zeros(N,1);
% for i=1:N
%     for j=1:N
%             kemeny(i)=kemeny(i)+Mmfpt(i,j)*pi_biased(j);
%     end
% end
% min(kemeny)
% max(kemeny)
