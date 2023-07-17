C_g(1:Ngauss) = v(1:Ngauss);
std_g(1:Ngauss) = v(Ngauss+1:2*Ngauss);
bias=zeros(N,1);
for i = 1:1:Ngauss
    bias(1:N) = bias(1:N) + amp * exp(-(xx' - C_g(i)).^2/(2*std_g(i)^2));
end
plot((1:N),bias,'--','Linewidth',1.5)
for i = 1:N-1
    u_ij=bias(i+1)-bias(i);
    K_biased(i,i+1)=K(i,i+1)*exp(-u_ij/2/kT);
    K_biased(i+1,i)=K(i+1,i)*exp(u_ij/2/kT);
    K_biased(i,i)=0;
end
K_biased(N,N)=0;
%% Normalizing the rate matrix.
for i = 1:N
    f=sum(K_biased(i,:));
    K_biased(i,i) =  -f;
end
%% biased MFPTs
M_t = expm(K_biased*t); % transition matrix.