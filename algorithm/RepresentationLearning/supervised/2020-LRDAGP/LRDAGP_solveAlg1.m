function [Z,E] = LRDAGP_solveAlg1(trainX,P,options)
%% Target
%%%     Solve Z and E
%% input
%%%     trainX              The training samples, m*n
%%%     P                   The learned projection matrix, d*m
%%%     options
%%%%%           mu          The lagrange coefficient
%%%%%           muMax       The maximum value of `mu`
%%%%%           rho         The increase rate of `mu`
%%%%%           t           The iteration times
%%%%%           beta        The weight of nuclear norm w.r.t J (Z)
%%%%%           theta       The weight of L2,1 norm w.r.t E
        %% Parameters
        mu=options.mu;
        muMax=options.muMax;
        rho=options.rho;
        T=options.t;
        beta=options.beta;
        theta=options.theta;
        epsilon=1e-8;
       %% Init
        [m,n]=size(trainX);
        Z=0;E=0;
        Y1=0;Y2=0;
        for i=1:T
            % Update J by Eq.(19)
            J=SVT(Z+Y2/mu,beta/mu);
            % Update Z by Eq.(20)
            Z=(trainX'*(P*P')*trainX+eye(n))\(trainX'*P*(P'*trainX-E+Y1/mu)-Y2/mu+J);
            % Update E by Eq.(21)
            E=SolveL21Problem(P'*trainX-P'*trainX*Z+Y1/mu,theta/mu);
            % Update Lagrange multipliers
            cd1=P'*trainX-P'*trainX*Z-E;
            cd2=Z-J;
            Y1=Y1+mu*(cd1);
            Y2=Y2+mu*(cd2);
            mu=min(rho*mu,muMax);
            if norm(cd1,'inf')<epsilon&&norm(cd2,'inf')<epsilon
                fprintf('[Z/E] is convergent at %d-th and break.\n',i); 
                break;
            end
        end
end

