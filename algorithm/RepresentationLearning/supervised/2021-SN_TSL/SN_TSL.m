function [results,results_iter,QW]=SN_TSL(trainX,trainY,testX,testY,options)
%% Implementation of SN_TSL
%%% Authors:                    Chen et al.
%%% Titl:                       2021-Sparse non-negative transition subspace learning for image classification
%% intput:
%%% trainX:                     The traing samples, m*n1
%%% trainY:                     The labels of training samples, n1*1
%%% testX:                      The test samples, m*n2
%%% testY:                      The labels of test samples, n2*1
%% options
%%% T                            The iterations
%%% alpha                        The weight of L1-norm w.r.t J (U)
%%% beta                         The weight of label regression
%%% lambda                       The regularization weight
%%% mu                           The Lagrange coefficient
%%% muMax                        The maximum value of mu
%%% rho                          The increase rate
%% output:
%%% results                      The results (list) [acc,NMI,purity]
%%% results_iter                 The iteration information of 'results'
%%% QW                            The learned projection matrix
%% Version
%%%     Implementation          2022-05-28
    options=defaultOptions(options,...
                'T',10,...              %%% The iterations
                'alpha',0.1,...        %%% The weight of L1-norm w.r.t J (U)
                'beta',1,...          %%% The weight of label regression
                'lambda',0.1,...          %%% The regularization weight
                'mu',1e-5,...            %%% The Lagrange coefficient
                'muMax',1e8,...         %%% The maximum value of mu
                'rho',1.1);            %%% The increase rate
    %% parameters
    T=options.T;
    alpha=options.alpha;
    beta=options.beta;
    lambda=options.lambda;
    mu=options.mu;
    rho=options.rho;
    muMax=options.muMax;

    %% Initialization
    results_iter=[];
    C=length(unique(trainY));
    [m,n1]=size(trainX);
    % Initialize P
    XX=trainX*trainX';
    % Init H
    H=hotmatrix(trainY,C,0)';
    Q=0;Y=0;J=0;
    W=H*trainX'*(XX'+lambda*eye(m));
    for i=1:T
        % Update Omega by Eqs.(15) and (16)
        Omega=((mu+1)*eye(C)+beta*Q*Q')\(W*trainX+beta*Q'*H+mu*J-Y);
        Omega=max(Omega,0);
        % Update J by Eq.(18)
        J=SVT(Omega+Y/mu,alpha/mu);
        % Update W by Eq.(12)
        W=(Omega*trainX')/(XX+lambda*eye(m));
        % Update Q by Eq.(14)
        Q=(beta*H*Omega')/(beta*(Omega*Omega')+lambda*eye(C));
        % Update Lagrange coefficient
        Y=Y+mu*(Omega-J);
        mu=min(rho*mu,muMax);
        % Classification
        QW=Q*W;
        Zs=QW*trainX; 
        Zt=QW*testX;
        Ytpseudo=classifyKNN(Zs,trainY,Zt,1);
        results=MyClusteringMeasure(testY,Ytpseudo,1);%[ACC MIhat Purity]';
        for index=1:3
            results_iter(index,i)=results(index);
        end
        fprintf('[%d]-th acc:%.4f, MIhat: %.4f, Purity:%.4f\n',i,...
            results(1),results(2),results(3));
    end
end