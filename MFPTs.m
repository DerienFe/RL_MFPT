function [MFPT] = MFPTs(i,j,p_eq,K)
% from i to j
% p_eq is the equilibrium probability 
% K is the rate matrix


one_row = ones(1,size(p_eq,2));
inverse = inv(p_eq * one_row - K);
MFPT = 1/p_eq(j)*(inverse(j,j) - inverse(j,i));


end