function Dij=fn_Euclidean_matrix_1Fr(xp,yp,S)
% copute a matrix of distances (NxN) for only current frame t
N=length(S);
Dij=double(-1*ones(N));% initialization, Dij(-1) means there is no a distance value computed between i, j

ID=1:N;
id=ID(S>0);
m=length(id); %  pedestrian in the current frame t

for i=1:m
    for j=i+1:m
        xi=xp(i); yi=yp(i);
        xj=xp(j); yj=yp(j);
        De=( sqrt((xi-xj).^2+(yi-yj).^2));
        Dij(id(i),id(j))=De;
    end
end

Dij=Dij+Dij'+1;

