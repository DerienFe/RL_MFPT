function[out] = gaussmf(X, [SIGMA, C])
%GAUSSMF Summary of this function goes here

  out = exp(-(X - C).^2/(2*SIGMA^2));
end

