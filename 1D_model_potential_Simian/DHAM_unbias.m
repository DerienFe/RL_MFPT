function[prob_dist] = DHAM_unbias(Record_states, x_eq, force_constant, kT, n_states, bias_order_win, cutoff)

qspace = linspace(0.9,n_states+1,n_states+1);
numsims=size(Record_states,1);
datlength(1:numsims)=size(Record_states,2);
numbins=n_states;

lagtime=1;
for k=1:numsims
        ncount(k,:)=histc(Record_states(k,1:datlength(k)-lagtime),qspace);
end

MM=zeros(numbins,numbins);
for k=1:numsims
        for i=1:datlength(k)
                [a,b(i)]=max(histc(Record_states(k,i),qspace));
        end
        for i=1+lagtime:datlength(k)
                msum=0;
                for l=1:numsims
                    nc=ncount(l,b(i-lagtime));
                    epot_j=0.5*force_constant*(bias_order_win(l,b(i))-x_eq(l))^2;
                    epot_i=0.5*force_constant*(bias_order_win(l,b(i-lagtime))-x_eq(l))^2;
                    delta_epot=max(-cutoff,epot_j-epot_i); %apply the cutoff here.
                    delta_epot=min(cutoff,delta_epot); %apply the cutoff here.
                    if (nc > 2 )
                        msum=msum+nc*exp(-delta_epot/kT/2);
                    end
                end
                if msum > 0 
                   MM(b(i-lagtime),b(i))=MM(b(i-lagtime),b(i))+1/msum;
                end
        end     
end     

msum=sum(MM');
for i=1:numbins
    if msum(i) > 0
        MM(i,:)=MM(i,:)/msum(i);
    else
        MM(i,:)=0;
    end
end

[v,d]=eig(MM');
[eigvals,l]=sort(diag(d),'ascend');
prob_dist=v(:,l(numbins))';
prob_dist=prob_dist/sum(prob_dist);
relax_time=-lagtime/log(eigvals(end-1));

end

