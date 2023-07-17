function[prob_dist,M0] = DHAM_sym(trajectory, bias_traj, kT, N, datlength,cutoff,amp)
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
sumtr=zeros(numbins,numbins);
trvec=zeros(numsims,numbins);
for k=1:numsims
        for i=1:datlength(k)
                [a,b(i)]=max(histc(data(k,i),qspace));
        end
        Ntr=zeros(numbins,numbins);
        for i=1+lagtime:datlength(k)
                Ntr(b(i-lagtime),b(i))=Ntr(b(i-lagtime),b(i))+1;
        end
        Ntr=(Ntr+Ntr')/2;
        for i=1:numbins
                for j=1:numbins
                        sumtr(i,j)=sumtr(i,j)+Ntr(i,j);
                        trvec(k,i)=trvec(k,i)+Ntr(i,j);
                end
        end
end

for a=1:numbins
    for b=1:numbins
      if sumtr(a,b) > 0
        sump1=0.0;
        for k=1:numsims
            epot_j=bias(k,b);
            epot_i=bias(k,a);
            delta_epot=max(-cutoff,epot_j-epot_i);
            delta_epot=min(cutoff,delta_epot);
            if trvec(k,a) > 0
                sump1=sump1+trvec(k,a)*exp(-(delta_epot)/2/kT);
            end
        end
            MM(a,b)=sumtr(a,b)/sump1;
      else
            MM(a,b)=0.;
      end
    end
end
msum=sum(MM');
% Normalize the Markov matrix
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
plot(mU2)
% hold on 
% plot(F)
%end

