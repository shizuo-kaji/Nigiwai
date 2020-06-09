%% this file computes Nigiwai score taking into account the Np prevoius frames
clear
clc
name='OnlyPassing'; % only change name of file  <<====================
signV=1;   % sign or absolute
Np=3;      % No of previus frames taken into account

vidObj = VideoWriter([ name '_Nigiwa.avi']);
vidObj.FrameRate=15;
open(vidObj);

Im0=imread([name '.png']);
load ([name '_Fulltrajectories'])




%---------------------------------------------------------
mask=Ytr>50&Ytr<350;  % take only ROI
Xtr(~mask)=0;
Ytr(~mask)=0;

[Nframes,N]=size(Xtr);
[Him,Wim]=size(Im0);
Wv=1;%sqrt(Him^2+Wim^2)/2;

NigMat=zeros(Nframes,N);
ID=1:N;
Dija0=-1*ones(N);   %initialization martix of moving average ditsances
%Dija= martix of moving average ditsances at time t 
%Dija0=martix of moving average ditsances at time t-1

Den=zeros(Nframes,1); % density of people in each frame
SNig=zeros(Nframes,1);% summation of Nigiwai in each frame




for f=Np:Nframes

    Sx=Xtr(f,:);
    idp=ID(Sx>0);
    NoPed=length(idp);  % current pedestrian
    
    if length(idp)>1
        xp=Xtr(f-Np+1:f,Sx>0);
        yp=Ytr(f-Np+1:f,Sx>0);
        Dija=fn_Euclidean_matrix_NpFrames(xp,yp,Sx);
        
        Nig=fn_Nigiwai_indicator(Dija,Dija0,Wv, signV); 
        snig=sum(Nig);
        SNig(f)=snig;%round(snig*1000)/1000;
        Den(f,1)=NoPed;
      
        Nig=round(Nig(Sx>0)*1000);
        

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

        
        

        Dija0=Dija;


    
    
    end




    
    
    
    
    
    
    
end
close(vidObj)





