function [gaussian] = gaussian_bias(x, C_g, std_g)
    amp = 1;
    gaussian = amp * exp(-(x - C_g).^2 / (2*std_g^2));
end