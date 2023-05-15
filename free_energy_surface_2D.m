
kT = 0.5981;
beta = 1./kT;
barrier_height = 2;

gamma=0.01;
U = @(x,y) 5-barrier_height*log( (exp(-(x+2).^2 - (y+2).^2)./gamma ) + ...
    (exp(-(x-2).^2 - (y-1).^2)./gamma ) + ...
    (exp(-(x+3).^2 - 5*(y-2).^2)./gamma ) );

X_lower_boundary =  -3.0;
X_upper_boundary =   3.0;

Y_lower_boundary =  -3.0;
Y_upper_boundary =   3.0;

x = linspace(X_lower_boundary,X_upper_boundary,10^3);
y = linspace(Y_lower_boundary,Y_upper_boundary,10^3);

[X,Y] = meshgrid(x,y);

Z = beta.*U(X,Y);
Z = Z - abs(min(Z(:)));
Z = Z - min(Z(:));

figure()
contourf(X,Y,Z,30,'LineStyle', 'none')
colorbar

% figure()
% surf(X,Y,Z,'LineStyle', 'none')
% colorbar

