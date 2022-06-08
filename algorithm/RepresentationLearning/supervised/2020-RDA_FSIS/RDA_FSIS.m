function [results,results_iter,Q]=RDA_FSIS(trainX,trainY,testX,testY,options)
%% Implementation of RDA_FSIS
%%% Authors                     Dornaika et al.
%%% Titl                        2020-Linear embedding by joint Robust Discriminant Analysis and Inter-class Sparsity
%% intput:
%%% trainX                      The traing samples, m*n1
%%% trainY                      The labels of training samples, n1*1
%%% testX                       The test samples, m*n2
%%% testY                       The labels of test samples, n2*1
%% options
%%% T                            The iterations
%%% dim                          The dimensions
%%% mu                           The weight of inter-class term in LDA (1e-4 in paper)
%%% lambda1                      The weight of L2,1 norm
%%% lambda2                      The weight of L1 norm in E
%%% lambda3                      The weight of L2-1 norm w.r.t Q'X
%%% beta                         The weight of Lagrange coefficient
%%% betaMax                      The maximum value of `beta`
%%% rho                          The increase rate
%%% epsilon                      The regularization terms
%% output:
%%% results                      The results (list) [acc,NMI,purity]
%%% results_iter                 The iteration information of 'results'
%%% Q                            The learned projection matrix
%% Version
%%%     Implementation          2022-06-08
    options=defaultOptions(options,...
                    'T',10,...          %%% The iterations
                    'dim',100,...       %%% The dimensions
                    'mu',1e-4,...       %%% The weight of inter-class term in LDA (1e-4 in paper)
                    'lambda1',1,...     %%% The weight of L2,1 norm w.r.t. Q
                    'lambda2',1,...     %%% The weight of L1 norm in E
                    'lambda3',1,...     %%% The weight of L2-1 norm w.r.t Q'X
                    'beta',1e-8,...     %%% The weight of reconstruction (1e-8 in paper)
                    'betaMax',1e5,...   %%% The maximum value of beta
                    'rho',10,...        %%% The increase rate
                    'epsilon',1);       %%% The regularization terms
    %% parameters
    T=options.T;
    dim=options.dim;
    beta=options.beta;
    mu=options.mu;
    lambda1=options.lambda1;
    lambda2=options.lambda2;
    lambda3=options.lambda3;
    rho=options.rho;
    betaMax=options.betaMax;
    epsilon=options.epsilon;
    %% Initialization
    results_iter=[];
    [m,n1]=size(trainX);
    X=[trainX,testX];
    n=size(X,2);
    D=eye(m,m);
    XX=trainX*trainX';
    E=zeros(m,n1);
    Sw=withinScatter(trainX,trainY);
    Sb=betweenScatter(trainX,trainY);
    Y1=zeros(m,n1);
    Y2=zeros(dim,n1);
    % Initialize P
    S=Sw-mu*Sb;
    S=1/n1*S;
%     left=left./norm(left,'fro');
    [P,~]=eigs(S+epsilon*eye(m),eye(m),dim,'sm');
    for i=1:T
        % Solve F by (24)
        if i==1
            F=0;
        else
            H=Q'*trainX;
            F=SolveL21Problem(H,lambda3/beta);
        end
        % Solve Q by Eq.(12)
        M=trainX-E+Y1/beta;
        Mprime=F+Y2/beta;
        Q=(2*(S)+lambda1*D+2*beta*XX)\(beta*(trainX*M'*P+trainX*Mprime'));
        % Solve P by Eq.(16)
%         [Up,~,Vp]=svd(M*trainX'*Q,'econ');
        [Up,~,Vp]=mySVD(M*trainX'*Q,dim);
        P=Up*Vp';
        % Solve E by Eq.(19)
        e=lambda2/beta;
        E=shrink(trainX-P*Q'*trainX+Y1/beta,e);
        % Update Lagrangian multiplier
        Y1=Y1+beta*(trainX-P*Q'*trainX-E);
        Y2=Y2+beta*(F-Q'*trainX);
        beta=min(rho*beta,betaMax);
        % Update D
        D=2*updateL21(Q);
        % Classification
        Z=Q'*X;
        Z=L2Norm(Z')';
        Zs=Z(:,1:n1);
        Zt=Z(:,n1+1:end);
        Ytpseudo=classifyKNN(Zs,trainY,Zt,1);
        results=MyClusteringMeasure(testY,Ytpseudo,1);%[ACC MIhat Purity]';
        for index=1:3
            results_iter(index,i)=results(index);
        end
        fprintf('[%d]-th acc:%.4f, MIhat: %.4f, Purity:%.4f\n',i,...
            results(1),results(2),results(3));
    end
end