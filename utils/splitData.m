function [X1,Y1,X2,Y2] = splitData(varargin)
% input: 
%   X:fea*n
%   Y:n*1
%   num: 0~1 or split number
% output 
%   X1: fea*num
%   Y1: num*1
%   X2: fea*(n-num)
%   Y2: (n-num)*1
    X1=[];X2=[];Y1=[];Y2=[];
    if nargin==3
        X=varargin{1};
        Y=varargin{2};
        num=varargin{3};
    else
         error("splitData: At least 3 parameters are required\n");
    end
    if num==0
         error("splitData: Parameter{3} must greater than zero\n");
    end
    if size(Y,2)>1
       Y=Y'; 
    end
    n=size(X,2);
    C=length(find(unique(Y)));
    if 0<num&&num<1
        num=max(1,floor(num*n));
    end
    %% split start
    for i=1:C
        Cn=length(find(Y==i));
        selectedNumber=min(max(floor(num),1),Cn);
        randomIndex=randperm(Cn);% random
        pos=find(Y==i);
        len=length(pos);
        if selectedNumber>len
            warning('selectedNumber (%d)> len (%d)!\n',selectedNumber,len);
            selectedNumber=len;
        end
        index=pos(randomIndex(1:selectedNumber));
        index2=pos(randomIndex(selectedNumber+1:end));
        X1=[X1,X(:,index)];Y1=[Y1;Y(index)];
        X2=[X2,X(:,index2)];Y2=[Y2;Y(index2)];
    end
end

