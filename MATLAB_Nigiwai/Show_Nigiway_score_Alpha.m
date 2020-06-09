% using (alpha and 1-Alpha) for compute Nigiwai
clear
clc
name='Passing_congestion'; % only change the name of file <<====================
Alpha=0.7;
signV=1;
vidObj = VideoWriter([ name '_alpha_nigiwai.avi']);
vidObj.FrameRate=15;
open(vidObj);

Im0=imread([name '.png']);
load ([name '_Fulltrajectories'])

%---------------------------------------------------------
mask=Ytr>50&Ytr<350; % ROI
Xtr(~mask)=0;
Ytr(~mask)=0;

[Nframes,N]=size(Xtr);
[Him,Wim]=size(Im0);
Wv=1;%sqrt(Him^2+Wim^2)/2;
NigMat=zeros(Nframes,N);
ID=1:N;
Dija0=-1*ones(N);      %initialization martix of moving average ditsances
%Dija= martix of moving average ditsances at time t 
%Dija0=martix of moving average ditsances at time t-1
Den=zeros(Nframes,1); % density of people in each frame
SNig=zeros(Nframes,1); % summation of Nigiwai in each frame


for f=1:Nframes

    Sx=Xtr(f,:);
    idp=ID(Sx>0);
    NoPed=length(idp);
    
    if length(idp)>1
        xp=Xtr(f,Sx>0);
        yp=Ytr(f,Sx>0);
        
        Dij=fn_Euclidean_matrix_1Fr(xp,yp,Sx); 
        Dija=Dij;
        mas=(Dij>-1)&(Dija0>-1);
        Dija(mas)=Alpha*Dij(mas)+(1-Alpha)*Dija0(mas); % apply equation of Alpha for only values of only existence distances (>-1)
  
        Nig=fn_Nigiwai_indicator(Dija,Dija0,Wv, signV); 
        Dija0=Dija;
        
        snig=sum(Nig);
        SNig(f)=snig;%round(snig*1000)/1000;
        Den(f,1)=NoPed;
        
%          Nig=Nig(Sx>0);
        Nig=round(Nig(Sx>0)*1000)/1000;
        

%         Nig=round(Nig*1000)/1000;
        xf=xp(end,:);
        yf=yp(end,:);
        L=length(xf);
        position=[xf' yf' 3*ones(L,1)];

        Im = insertObjectAnnotation(Im0, 'circle', position,Nig,'Color','red','TextBoxOpacity',0,'FontSize',8);
        Im = insertText(Im, [10 10], ['Frame=' mat2str(f) ],'FontSize',10);
        Im = insertText(Im, [175 10], ['Ped=' mat2str(NoPed)],'FontSize',10  );
        Im = insertText(Im, [175 30], ['Nigiwai=' mat2str(snig)],'FontSize',10 );
%         imshow(Im);
%         pause(0.04)
     writeVideo(vidObj,Im);

    end

end
close(vidObj)




