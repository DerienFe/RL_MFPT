for i=1:Ngauss
    C_g(i)=rand+current_pos;
    std_g(i) = rand*25;
end
bias=zeros(N,1);
for i = 1:1:Ngauss
    bias(1:N) = bias(1:N) + amp * exp(-(xx' - C_g(i)).^2/(2*std_g(i)^2));
end
figure
plot(xx,F,'b', 'LineWidth', 2)
hold on
plot(xx,bias, 'r', 'LineWidth', 2)
plot(xx,bias+F'-min(bias+F'), 'g--', 'LineWidth', 2)