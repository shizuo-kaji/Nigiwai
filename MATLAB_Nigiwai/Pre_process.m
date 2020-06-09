
%% this file read trajectories from Vadere format and complete the miss values by repetition
% finally save Xtr, Ytr in the form of MxN, M=number of frames,N= No of pedestrain
% Xtr, Ytr represent x and y values respecively 
% M=number of frames ;N= No of pedestrain; H=hight ;scale=scale from VADERE size to video size

clear
clc

name='OnlyPassing'; % only change the name of image.
Im=imread([name '.png']);
scale=10; 
T = readtable([name '.txt']); % read trajectories file
H=size(Im,1);

G0 = table2array(T);
G=G0;
G(:,2)=round(G(:,2)*2.5)+1; % convert time step into integer= frame No, start from 1
Gs = sortrows(G,1);
id=Gs(:,1); % pedestrain ID
F=Gs(:,2);  % frame No
N=max(id); %No of pedestrain
M=max(F); %number of frames
FullTr=zeros(M*N,size(Gs,2));

h=hist(id,1:max(id)); % determine the length of each ID (NO of frames for each pedestrian in which he exist)

s1=1;
for i=1:N   %No of pedestrain
    s2=s1+h(i)-1;
    Gi=Gs(s1:s2,:); % data for pedestrain i
    A=fn_repeat_frames(Gi,M); % complete the missed values on trajectories by repetition
    FullTr(M*(i-1)+1:M*i,:)=A;       
    s1=s2+1;
end    

  
FullTrSorted = sortrows(FullTr,2);
 [Xtr,Ytr]=fn_FormatTrajectory(FullTrSorted, M,N,H,scale);
 
save ([name '_Fulltrajectories.mat'], 'FullTr', 'FullTrSorted', 'Xtr', 'Ytr')
% save Simple_Fulltrajectories FullTr FullTrSorted Xtr Ytr 
% save Supermarket_Fulltrajectories FullTr FullTrSorted Xtr Ytr












