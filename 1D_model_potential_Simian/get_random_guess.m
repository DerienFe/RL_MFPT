pi_biased = zeros(1, N);
K_biased = zeros(size(K));
mfpt = inf;
bias_opt = zeros(N,1);
for Nopt=1:1
    for i=1:Ngauss
        C_g(i)=rand*100;
        std_g(i) = rand*25;
    end
    bias=zeros(N,1);
    for i = 1:1:Ngauss
        bias(1:N) = bias(1:N) + amp * exp(-(xx' - C_g(i)).^2/(2*std_g(i)^2));
    end
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
    [pi_biased, ~] = compute_free_energy(K_biased', kT);
    [mfpts] = mfpt_calc(pi_biased,K_biased);
    mfpt_mid = mfpts(state_start, state_end);
    M_t = expm(K_biased*t); % transition matrix.
    Mmfpt=Markov_mfpt_calc(pi_biased',M_t);
    mfpt_mid = Mmfpt(state_start, state_end)*t;
    if mfpt_mid < mfpt
        mfpt = mfpt_mid;
        bias_opt = bias;
        v(1:Ngauss)=C_g(1:Ngauss);
        v(Ngauss+1:2*Ngauss)=std_g(1:Ngauss);
    end
end
figure
plot(xx,F,'b', 'LineWidth', 2)
hold on
plot(xx,bias_opt, 'r', 'LineWidth', 2)
plot(xx,bias_opt+F'-min(bias_opt+F'), 'g--', 'LineWidth', 2)