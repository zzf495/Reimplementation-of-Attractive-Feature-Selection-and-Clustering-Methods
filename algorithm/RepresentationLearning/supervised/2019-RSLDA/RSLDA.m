function [results,results_iter,Q]=RSLDA(trainX,trainY,testX,testY,options)
%% Implementation of RSLDA
%%% Authors:                    Wen et al.
%%% Titl:                       2019-Robust Sparse Linear Discriminant Analysis
%% intput:
%%% trainX:                     The traing samples, m*n1
%%% trainY:                     The labels of training samples, n1*1
%%% testX:                      The test samples, m*n2
%%% testY:                      The labels of test samples, n2*1
%% options
%%% T                            The iterations
%%% dim                          The dimensions
%%% mu                           The weight of inter-class term in LDA (1e-4 in paper)
%%% lambda1                      The weight of L2,1 norm
%%% lambda2                      The weight of L1 norm in E
%%% betaMax                      The maximum value of beta
%%% beta                         The weight of reconstruct (ADMM)
%%% rho                          The increase rate
%%% epsilon                      The regularization terms
%% output:
%%% results                      The results (list) [acc,NMI,purity]
%%% results_iter                 The iteration information of 'results'
%%% Q                            The learned projection matrix
%% Version
%%%     Implementation          2022-05-18
    options=defaultOptions(options,...
                'T',10,...          %%% The iterations
                'dim',10,...       %%% The dimensions
                'mu',1e-4,...       %%% The weight of inter-class term in LDA (1e-4 in paper)
                'beta',0.1,...        %%% The weight of reconstruction (0.1 in paper)
                'lambda1',1,...     %%% The weight of L2,1 norm
                'lambda2',1e-3,...   %%% The weight of L1 norm in E
                'betaMax',1e5,...   %%% The maximum value of beta
                'rho',1.01,...      %%% The increase rate
                'epsilon',1);     %%% The regularization terms
    %% parameters
    T=options.T;
    dim=options.dim;
    beta=options.beta;
    mu=options.mu;
    lambda1=options.lambda1;
    lambda2=options.lambda2;
    rho=options.rho;
    betaMax=options.betaMax;
    epsilon=options.epsilon;

    %% Initialization
    results_iter=[];
    [m,n1]=size(trainX);
    X=[trainX,testX];
    D=eye(m,m);
    XX=trainX*trainX';
    E=zeros(m,n1);
    Sw=withinScatter(trainX,trainY);
    Sb=betweenScatter(trainX,trainY);
    
    Lagrangian=zeros(m,n1);
    % Initialize P
    left=Sw-mu*Sb;
    left=1/n1*left;
%     left=left./norm(left,'fro');
    [P,~]=eigs(left+epsilon*eye(m),eye(m),dim,'sm');
    for i=1:T
        % Solve Q by Eq.(15)
        M=trainX-E+Lagrangian/beta;
        Q=(2*(left)+lambda1*D+beta*XX)\(beta*trainX*M'*P);
        % Solve P by Eq.(16)
%         [Up,~,Vp]=svd(M*trainX'*Q,'econ');
        [Up,~,Vp]=mySVD(M*trainX'*Q,dim);
        P=Up*Vp';
        % Solve E by Eq.(19)
        e=lambda2/beta;
        E=shrink(trainX-P*Q'*trainX+Lagrangian/beta,e);
        % Update Lagrangian multiplier
        Lagrangian=Lagrangian+beta*(trainX-P*Q'*trainX-E);
        beta=min(rho*beta,betaMax);
        % Update D
        D=2*updateL21(Q);
        % Classification
        Z=Q'*X;
        Z=L2Norm(Z')';
        Zs=Z(:,1:n1);
        Zt=Z(:,n1+1:end);
        Ytpseudo=classifyKNN(Zs,trainY,Zt,1);
        results=MyClusteringMeasure(testY,Ytpseudo,1);%[ACC ACC2 MIhat Purity]';
        for index=1:3
            results_iter(index,i)=results(index);
        end
        fprintf('[%d]-th acc:%.4f, MIhat: %.4f, Purity:%.4f\n',i,...
            results(1),results(2),results(3));
    end
end