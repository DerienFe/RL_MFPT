function [mfpt] = mfpt_dat(traj,N,lagtime)

	datlength=size(traj,2);
%%
%
maxlength=max(datlength);
%mfpt_k=zeros(numsims,N,N);
        for a=1:N
            g=find(traj==a);
            sg=size(g,2);
            inds(a,1:sg)=g;
            size_a(a)=sg;
        end
        mfpt=NaN(N,N);
        for a=min(traj):max(traj)
            s_a=size_a(a);
            mfpt_i=NaN(s_a,N);
            for i=1:s_a
                init=inds(a,i);
                for b=min(traj):max(traj)
                    dum=inds(b,1:size_a(b));
                    dum(dum<init)=NaN;
                    mfpt_i(i,b)=min(dum)-init;
                end
            end
            mfpt(a,:)=nanmean(mfpt_i,1);
        end
        mfpt=mfpt*lagtime;
%mfpt=squeeze(nanmean(mfpt_k,1));

end
