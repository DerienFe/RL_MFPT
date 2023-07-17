function[K] = create_K_1D(N, kT)


x=linspace(0,5*pi,N);
y1=sin((x-pi));
y2=sin((x-pi)/2);
A=10;
xtilt=0.5;
y=xtilt*y1-y2;
y=xtilt*y1+(1-xtilt)*y2;
y=y*3;  
for i=1:N-1
    K(i,i+1)=A*exp((y(i+1)-y(i))/2/kT);
    K(i+1,i)=A*exp((y(i)-y(i+1))/2/kT);
end

for i=1:N
    K(i,i)=0;
    K(i,i)=-sum(K(:,i));
end
K=K';
end