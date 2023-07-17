function [mfpt] = mfpt_calc(peq,K)
% Computes mean first passage times
%   Inputs: 
%   peq = equilibrium population 
%   K = Rate Matrix
%   N = Number of states 
N=size(K,1);
onevec=ones(N,1);
A=peq'*onevec';
A=A';
Qinv=inv(A-K);
mfpt=zeros(N,N);
for j=1:N
    for i=1:N
        mfpt(i,j)=1/peq(j)*(Qinv(j,j)-Qinv(i,j));
    end
end
