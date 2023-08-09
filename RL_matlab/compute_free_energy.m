% function which given a Markov model T calculates the equilibrium probabilities eq
function [pi, F, eigenvectors, eigenvalues, eigenvalues_sorted, index]=compute_free_energy(K, kT)
%%
beta = 1./kT;
[eigenvectors,eigenvalues] = eig(K); % compute the eigenvalues and eigenvectors. 
[eigenvalues_sorted,index] = sort(diag(eigenvalues),'descend'); % sort the eigenvalues in descending order.
pi = eigenvectors(:,index(1))'/sum(eigenvectors(:,index(1))); % The stationary distribution corresponds to the eigenvector corresponding to the largest eigenvalue.
F =-log(pi)./(beta); % Free energy. 

end