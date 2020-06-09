function [Xtr,Ytr]=fn_FormatTrajectory(FullTrSorted, M,N,H,scale)

% make trajectoris in the form of MxN, M=number of frames,N= No of pedestrain
% Xtr, Ytr represent x and y values respecively 
% FullTrSorted
%
Xtr=zeros(M,N);
Ytr=zeros(M,N);
G=FullTrSorted;
X=G(:,4)*scale; % scale
Y=G(:,5)*scale;
Y=H-Y;      % y in VADERE is oppisit in MATLAB 

for i=1:M
    Xtr(i,:)=(X((i-1)*N+1 : N*i ))';
    Ytr(i,:)=(Y((i-1)*N+1 : N*i ))';
 
end


