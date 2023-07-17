function[bias] = predict_bias(M0,v0,cutoff,Ngauss,amp,kT,current_pos,state_end,tot_time)
%%
if tot_time==0
    bias=v0; %zeros(1,2*Ngauss);
else
    N = size(M0,2);
%     M1=M0;
%     for i=1:N
%         M1(i,i)=0;
%     end
    xx=(1:N);
    working_ind=(sum(M0')>0);
    wx=xx(working_ind);
    [ff,gg]=max(wx==current_pos);
    wM=M0(sum(M0')>0,sum(M0')>0);
    %[ff,gg2]=max(wx==state_end);  Needs small improvement
    % gg2=max(state_end,gg2);
    target_state=min(size(wx,2),state_end) % sub with gg2
    LB = ones(1, Ngauss); % lower bound
    UB = repmat(N, 1, Ngauss); % upper bound
    [bias,b,c]= fmincon(@(v) min_M_mfpt(v,wx,wM,cutoff,Ngauss,amp,kT,gg,target_state), v0,[],[],[],[],LB,UB);
    %[bias,b,c]=fminsearch(@(v) min_M_mfpt(v,wx,wM,cutoff,Ngauss,amp,kT,gg,target_state),v0);%,optimset('MaxFunEvals',10000));
    b
end
