function Dij=fn_Euclidean_matrix_NpFrames(xp,yp,S)
% copute a matrix of distances (NxN) taking into account the previous Np frames 
Np=size(xp,1);%N= no of pervius frames
N=length(S); % no of total pedestrians
Dij=-1*ones(N); % initialization, Dij(-1) means there is no a distance value computed between i, j
k=0:Np-1;
wk=1./2.^k;
ID=1:N;
id=ID(S>0);
m=length(id);
% we compute only the upper right elements of matrix then adding it to the transpose one 
for i=1:m
    for j=i+1:m
        xi=xp(:,i); yi=yp(:,i);
        xj=xp(:,j); yj=yp(:,j);
        m1=(xi>0);
        m2=(xj>0);
        mc=m1 & m2 ; % only common points
%         sum(mc)
        xi=xi(mc);
        yi=yi(mc);
        xj=xj(mc);
        yj=yj(mc);
        wkc=wk(mc);
        Swk=sum(wkc);
               
        De=( sqrt((xi-xj).^2+(yi-yj).^2));
        
        dij=sum(De'.*wkc)/Swk;
        Dij(id(i),id(j))=dij;
 
    end
end
Dij=Dij+Dij'+1; % add 1 for cancel effect of -1
