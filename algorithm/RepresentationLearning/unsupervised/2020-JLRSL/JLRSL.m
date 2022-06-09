function [results,results_iter,A]=JLRSL(trainX,trainY,testX,testY,options)
%% Notice
%%% The `trainY` is used to generate a 1NN classifier as described in the
%%% paper, while `testX` and `testY` are used to verify the effectiveness
%%% of JLRSL
%% Implementation of JLRSL
%%% Authors                     Peng et al.
%%% Titl                        2020-Joint low-rank representation and spectral regression for robust subspace learning
%% intput
%%% trainX                       The traing samples, m*n1
%%% trainY                       The labels of training samples, n1*1
%%% testX                        The test samples, m*n2
%%% testY                        The labels of test samples, n2*1
%% options
%%% T                            The iterations
%%% dim                          The dimensions
%%% lambda1                      The weight of nuclear norm w.r.t J (Z)
%%% lambda2                      The weight of L2,1 norm w.r.t E
%%% lambda3                      The weight of F2 norm w.r.t A
%%% mu                           The Lagrange coefficient 
%%%                                     (default 1e-8 in the paper)
%%% muMax                        The maximum value of `mu` 
%%% rho                          The increase rate of Lagrange coefficient 
%%%                                     (default 1.1 in the paper)
%% output:
%%% results                      The results (list) [acc,NMI,purity]
%%% results_iter                 The iteration information of 'results'
%%% A                            The learned projection matrix
%% Version
%%%     Implementation          2022-06-09
    options=defaultOptions(options,...
                    'T',10,...              %%% The iterations
                    'dim',100,...           %%% The dimensions
                    'lambda1',1e3,...       %%% The weight of nuclear norm w.r.t J (Z)
                    'lambda2',1e1,...       %%% The weight of L2,1 norm w.r.t E
                    'lambda3',1e3,...       %%% The weight of F2 norm w.r.t A
                    'mu',1e-8,...           %%% The Lagrange coefficient 
                    'muMax',1e10,...        %%% The maximum value of `mu`
                    'rho',1.1);             %%% The increase rate of Lagrange coefficient 
    %% parameters
    T=options.T;
    dim=options.dim;
    lambda1=options.lambda1;
    lambda2=options.lambda2;
    lambda3=options.lambda3;
    rho=options.rho;
    mu=options.mu;
    muMax=options.muMax;
    %% Initialization
    results_iter=[];
    [m,n]=size(trainX);
    XTX=trainX'*trainX;
    Y1=zeros(m,n);
    Y2=zeros(n,n);
    E=zeros(m,n);
    J=zeros(n,n);
    Z=zeros(n,n);
    %% Init Y by Eq.(1)
    manifold.k = 0;
    manifold.Metric = 'Euclidean';
    manifold.WeightMode = 'HeatKernel';
    manifold.NeighborMode = 'KNN';
    manifold.t=mean(mean(EuDist2(trainX')));
    W=lapgraph(trainX',manifold);
    D=diag(sparse(sum(W)));
    [Y,~]=eigs(W,D,dim,'lm');
    for i=1:T
        % Solve A by Eq.(11)
        XZ=trainX*Z;
        left=XZ*XZ'+lambda3*eye(m); % m * m
        right=XZ*Y; % m * C
        A=left\right; % m * C
        % Solve Z by Eq.(13)
        left=trainX'*(A*A'+mu*eye(m))*trainX+mu*eye(n);  % n * n 
        right=(trainX'*A*Y'+trainX'*Y1-Y2+mu*(XTX-trainX'*E+J)); % n * n
        Z=left\right; % n * n
        % Solve J by Eq.(14)
        J=SVT(Z+Y2/mu,lambda1/mu); % n * n
        % Solve E by Eq.(15)
        E=SolveL21Problem(trainX-trainX*Z+Y1/mu,lambda2/mu); % m * n
        % Update Lagrange coefficient
        loss1=trainX-trainX*Z-E;
        loss2=Z-J;
        Y1=Y1+mu*loss1;
        Y2=Y2+mu*loss2;
        mu=max(rho*mu,muMax);
        % Clustering
        absZ=abs(Z); % n * n
        W=(absZ+absZ')/2;
        D=diag(sparse(sum(W)));
        [Y,~]=eigs(W,D,dim,'lm');
        %% Classification
        Zs=A'*trainX;
        Zt=A'*testX;
        Ypseudo=classifyKNN(Zs,trainY,Zt,1);
        results=MyClusteringMeasure(testY,Ypseudo);% [ACC MIhat Purity]
%         [~,results] = getClusteringResults(X*Z,realY,A);
        for index=1:3
            results_iter(index,i)=results(index);
        end
        fprintf('[%d]-th acc:%.4f, MIhat: %.4f, Purity:%.4f, loss1: %.6f, loss2: %.6f\n',i,...
            results(1),results(2),results(3),norm(loss1,'inf'),norm(loss2,'inf'));
    end
end