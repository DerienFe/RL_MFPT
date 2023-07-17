function[prob_dist,M0] = DHAM_unbias(trajectory, bias_traj, kT, N, datlength,cutoff,amp)
%%
qspace = linspace(0.9,N+1,N+1);
numsims=size(datlength,2);
numbins=N;
j=0;
for i=1:numsims
    dl=datlength(i);
    data(i,1:dl)=trajectory(j+1:j+dl);
    bias(i,1:numbins)=get_bias(bias_traj{i},numbins,amp);
    j=j+dl;
end
%%
lagtime=1;
%ncount=zeros(numsims,numbins+1);
for k=1:numsims
        ncount(k,:)=histc(data(k,1:datlength(k)-lagtime),qspace);
end
%%
MM=zeros(numbins,numbins);
for k=1:numsims
        for i=1:datlength(k)
                [a,b(i)]=max(histc(data(k,i),qspace));
        end
        for i=1+lagtime:datlength(k)
                msum=0;
                for l=1:numsims
                    nc=ncount(l,b(i-lagtime));
                    epot_j=bias(l,b(i));
                    epot_i=bias(l,b(i-lagtime));
                    delta_epot=max(-cutoff,epot_j-epot_i);
                    delta_epot=min(cutoff,delta_epot);
                    if (nc > 3)
                        msum=msum+nc*exp(-delta_epot/kT/2);
                    end
                end
                if msum > 0 
                   MM(b(i-lagtime),b(i))=MM(b(i-lagtime),b(i))+1/msum;
                end
        end     
end     
%%
msum=sum(MM');
for i=1:numbins
    if msum(i) > 0
        MM(i,:)=MM(i,:)/msum(i);
    else
        MM(i,:)=0;
    end
end
M0=MM;
%%
[v,d]=eig(MM');
[eigvals,l]=sort(diag(d),'ascend');
prob_dist=v(:,l(numbins))';
prob_dist=prob_dist/sum(prob_dist);
relax_time=-lagtime/log(eigvals(end-1));
mU2 = real(-kT*log(prob_dist));
mU2 = mU2 - min(mU2(1:20));
max(mU2);
% figure
plot(mU2,'LineWidth',2.5)
% hold on 
% plot(F)
%end

