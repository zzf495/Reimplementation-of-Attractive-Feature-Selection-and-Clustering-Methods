function [P] = LRDAGP_solveAlg2(X,P,Z,XLX,G,V,options)
%% Target
%%%     Solve P
%% input
%%%     X                   The training samples, m*n
%%%     P                   The learned projection matrix, d*m
%%%     Z                   The reconstruction matrix, n*n
%%%     XLX                 The manifold term of `X`
%%%     G                   The within-class scatter of `X`
%%%     V                   The between-class scatter of `X`
%%%     options
%%%%%           t           The iteration times
%%%%%           dim         The dimension reduced
%%%%%           alpha       The relative weight of Sb, i.e., Sw-alpha*Sb
%%%%%           lambda      The weight of scatters w.r.t P
%%%%%           theta       The weight of L2,1 norm w.r.t E
        %% Parameters
        T=options.t;
        lambda=options.lambda;
        alpha=options.alpha;
        theta=options.theta;
        dim=options.dim;
        epsilon=1e-8;
        myeps=1e-1;
        %% Init
        [m,n]=size(X);
        XZ=X*Z;
        IZ=eye(n,n)-Z;
        GV=lambda*G-alpha*V;
        U=eye(n);
        for i=1:T
            
            % Update P by Eq.(25)
            [P,~]=eigs(XLX+lambda*GV+theta*(X-XZ)*U*(X-XZ)'+myeps*eye(m),eye(m),dim,'sm');
            P=real(P);
            % Update U by Eq.(24);
            PX=P'*X;
            PXZ=(PX*IZ)';
            PXZ(PXZ==0)=myeps;
            U=updateL21(PXZ);
            if i>=2&&(norm(lastU-U,'fro')<epsilon)
                fprintf('[P] is convergent at %d-th and break.\n',i); 
                break;
            else
                lastU=U;
            end
        end
end

