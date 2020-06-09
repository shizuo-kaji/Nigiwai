function Nig=fn_Nigiwai_indicator(Dija,Dija0,Wv, signV)
%Dija= martix of moving average ditsances at time t 
%Dija0=martix of moving average ditsances at time t-1
N=size(Dija,1);
Nig=zeros(1,N);

for i=1:N
    Dj=Dija(:,i);
    Dj0=Dija0(:,i);

    if ~isempty(Dj(Dj>-1))
        id1=Dj>=0;
        id2=id1&Dj0<0; % in this case Dj0=-1,it should not calculate, so we make it zero to not affect 
        Dj0(id2)=0 ;   
        
        if signV==1
        delta= Dj(id1)-Dj0(id1);
        else
        delta= abs(Dj(id1)-Dj0(id1));
        end
        
        Dj2=  (Dj(id1)+1).^2;
        
        Nig(i)=sum( exp(-delta/Wv)  ./Dj2);

    end

end