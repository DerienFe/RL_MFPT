%here is the function helps create gaussian in 2d.

function [gaussian] = gaussian2d(x,y,x0,y0,sigma_x, sigma_y)
    gaussian = exp(-((x-x0).^2/(2*sigma_x^2)+(y-y0).^2)/(2*sigma_y^2));
end
