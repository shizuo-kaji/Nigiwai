function A=fn_repeat_frames(Gi,M)
%% complete missed values, put zeros in case first frame, otherwise repeat the prevoius ones
A=zeros(M,size(Gi,2)); % new matrix contains data in  all frames(M)
c=1;

for i=1:M
    if c<=size(Gi,1)
        if Gi(c,2)==i % value is exist at frame i
            A(i,:)=Gi(c,:);
            c=c+1;
        else
            if i>1
            A(i,:)= A(i-1,:);
            A(i,2)=i;
            else
            A(i,1)=Gi(1,1); % in case first frame all value are zeros except frame and ID numbers
            A(i,2)=1; % frame number

            end
        end
    else
A(i,:)= A(i-1,:);
A(i,2)=i; % % frame number  
    end
    
end






