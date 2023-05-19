function [mfpt] = mfpt_calc(peq,K,N)
% Computes Kemeny Gradient and Adam expression and compares the two
%   Inputs: 
%   peq = equilibrium population 
%   K = Rate Matrix
%   N = Number of states 

onevec=ones(N,1);
Qinv=inv(peq*onevec'-K');
%mfpt=zeros(N,N);
for j=1:N
    for i=1:N
        mfpt(i,j)=1/peq(j)*(Qinv(j,j)-Qinv(i,j));
    end
end
