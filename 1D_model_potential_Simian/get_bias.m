function[bias] = get_bias(v,N,amp)
%%
xx=(1:N);
Ngauss=size(v,2)/2;
C_g(1:Ngauss) = v(1:Ngauss);
std_g(1:Ngauss) = v(Ngauss+1:Ngauss+Ngauss);
bias=zeros(N,1);
for i = 1:1:Ngauss
    bias(1:N) = bias(1:N) + amp * exp(-(xx' - C_g(i)).^2/(2*std_g(i)^2));
end
