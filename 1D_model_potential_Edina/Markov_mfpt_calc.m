function [mfpt] = Markov_mfpt_calc(peq,M)
%   peq = equilibrium population 
%   M = Markov Matrix
%   N = Number of states 
N=size(M,1);
onevec=ones(N,1);
I=diag(onevec);
A=peq*onevec';
A=A';
Qinv=inv(I+A-M);
mfpt=zeros(N,N);
for j=1:N
    for i=1:N
        mfpt(i,j)=1/peq(j)*(Qinv(j,j)-Qinv(i,j)+I(i,j));
    end
    %mfpt(j,j)=0;
end
mfpt=mfpt;