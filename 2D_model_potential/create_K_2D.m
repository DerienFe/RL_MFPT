function[K,Z] = create_K_2D(N, kT, barrier_height)

A=10;
KbT=kT;
ngrid=N; %13
gamma=0.01;
[x,y]=meshgrid(linspace(-3,3,ngrid),linspace(-3,3,ngrid));

Z = 5-barrier_height*log( (exp(-(x+2).^2 - 3*(y+2).^2)./gamma ) + ...
    (exp(-5*(x-2).^2 - (y-2).^2)./gamma ) + ...
    (exp(-6*(x-2).^2 - 5*(y+2).^2)./gamma ) + ...
    (exp(-3*(x+2).^2 - (y-2).^2)./gamma ) );

for i=1:ngrid-1
    for j=1:ngrid
        nn=i+ngrid*(j-1);
        K(nn,nn+1)=A*exp((Z(i+1,j)-Z(i,j))/2/KbT);
        K(nn+1,nn)=A*exp((Z(i,j)-Z(i+1,j))/2/KbT);
    end
end

for i=1:ngrid
    for j=1:ngrid-1
        nn=i+ngrid*(j-1);
        K(nn,nn+ngrid)=A*exp((Z(i,j+1)-Z(i,j))/2/KbT);
        K(nn+ngrid,nn)=A*exp((Z(i,j)-Z(i,j+1))/2/KbT);
    end
end

NN=ngrid*ngrid;
for i=1:NN
    %K(i,i)=0;
    K(i,i)=-sum(K(:,i));
end

figure1 = figure
axes('FontSize',18);
hold on
[C,h] = contour(y,x,Z-min(Z(1:end)),'LevelStep',0.5,'Fill','on');
title('initial profile');
colorbar
xlabel(['x'],'FontSize',18);
ylabel(['y'],'FontSize',18);

Z=Z-min(Z(1:end));
end