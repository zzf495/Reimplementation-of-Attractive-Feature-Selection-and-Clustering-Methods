function [R] = MRSL_getR(F,pseudoLabel)
%% input:
%%%     F:              The regression matrix with dimensions C*n
%%%     pseudoLabel:    The pseudo labels of samples, n*1
%% output:
%%%     R:              The regression matrix solved, C*n
    [C,n]=size(F);
%     [~,pseudoLabel]=max(F,[],1);
    R=zeros(C,n);
    for idx=1:n
        m=pseudoLabel(idx);
        xi=0;t=0;
        Zj=F(:,idx)+1-repmat(F(m,idx),C,1);
        for c=1:C
            if m~=c
                zj=Zj(c);
                phiXi=2*xi+sum(min(xi-zj,0));
                if phiXi>0
                    xi=xi+Zj;
                    t=t+1;
                end
            end
        end
        xi=xi/(1+t);
        R(:,idx)=F(:,idx)+min(xi-Zj,0);
        R(m,idx)=F(m,idx)+xi;
    end
end

