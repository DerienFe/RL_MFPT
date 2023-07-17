function [MD]=jjhunter(P)
format long
m=length(P);
PS=single(P);
eS=eye(m);
e1S=ones(m,1);
ES=ones(m,m);
IS=eye(m);
PPS=PS;
AASS=zeros(m,m);
for n=m:-1:2
    SS(1,n)=sum(PPS(n,1:n-1));
    for i=1:n-1
        for j=1:n-1
            AASS(i,j)=PPS(i,n)*PPS(n,j)/SS(1,n);
            PPS(i,j)=PPS(i,j)+AASS(i,j);
        end
    end
end
rS=zeros(1,m);
rS(1,1)=1;
for n=2:m
    for i=1:n-1
        rS(1,n)=rS(1,n)+rS(1,i)*PPS(i,n)/SS(1,n);
    end
end
TOTS=sum(rS);
pit_1S=rS/TOTS;
PiS=e1S*pit_1S;
ZS=inv(IS-PS+PiS);
DS=inv(diag(diag(PiS)));
MS=(IS-ZS+ES*diag(diag(ZS)))*DS;

PD=double(P);
eD=[eye(m)];
e1D=ones(m,1);
ED=ones(m,m);
ID=eye(m);
PPD=PD;
AASD=zeros(m,m);
for n=m:-1:2
    SD(1,n)=sum(PPD(n,1:n-1));
    for i=1:n-1
        for j=1:n-1
            AASD(i,j)=PPD(i,n)*PPD(n,j)/SD(1,n);
            PPD(i,j)=PPD(i,j)+AASD(i,j);
        end
    end
end
rD=zeros(1,m);
rD(1,1)=1;
for n=2:m
    for i=1:n-1
        rD(1,n)=rD(1,n)+rD(1,i)*PPD(i,n)/SD(1,n);
    end
end
TOTD=sum(rD);
pit_1D=rD/TOTD;
PiD=e1D*pit_1D;
ZD=inv(ID-PD+PiD);
DD=inv(diag(diag(PiD)));
MD=(ID-ZD+ED*diag(diag(ZD)))*DD;