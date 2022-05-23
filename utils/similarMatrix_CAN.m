function [S,gamma] = similarMatrix_CAN(distP2D,k,rr)
%% input:
%%% distP2D:   the distance from data to prototype, n2*n1
%%% k:         the number of neighbors
%%% rr:        the optimal parameter
%%% lambda:    default = rr, and update ^2 /2 by eigs
%% output:
%%% S:         the similar matrix n2*n1
    [n2,n1]=size(distP2D);
%     distP2D=EuDist2(data',prototype); % n2*n1
    if nargin==2
       rr=-1; 
    end
    if rr == -1
        [d, idx] = sort(distP2D,2,'ascend');
        
        gamma=1/n2*sum( k/2* d(:,k+1) - 1/2* sum(d(:,1:k),2) );
        S = zeros(n2,n1);
        for i = 1:n2
            idxa0 = idx(i,1:k);
            S(i,idxa0)=EProjSimplex_new((d(i,k+1)-d(i,1:k))./(2*gamma));
        end
    else
        gamma=rr;
        [dx, idx] = sort(distP2D,2,'ascend');
        S = zeros(n2,n1);
        for i = 1:n2
            idxa0 = idx(i,1:k);
            S(i,idxa0)=EProjSimplex_new(-1*dx(i,1:k)./(2*gamma));
        end
    end
    
end

