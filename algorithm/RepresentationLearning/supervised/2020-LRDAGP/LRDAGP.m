function [results,results_iter,P]=LRDAGP(trainX,trainY,testX,testY,options)
%% Implementation of LRDAGP
%%%             Authors         Du et al.
%%%             Title           2020-Low-Rank Discriminative Adaptive Graph Preserving Subspace Learning
%% intput:
%%%             trainX          The traing samples, m*n1
%%%             trainY          The labels of training samples, n1*1
%%%             testX           The test samples, m*n2
%%%             testY           The labels of test samples, n2*1
%% options
%%%             T               The total iteration times
%%%             t               The iteration times
%%%             dim             The dimension reduced
%%%             k               The number of KNN
%%%             alpha           The relative weight of Sb, i.e., Sw-alpha*Sb
%%%             beta            The weight of nuclear norm w.r.t J (Z)
%%%             theta           The weight of L2,1 norm w.r.t E
%%%             lambda          The weight of scatters w.r.t P
%%%             mu              The lagrange coefficient
%%%             muMax           The maximum value of `mu`
%%%             rho             The increase rate of `mu`
%% output:
%%%             results         The results (list) [acc,NMI,purity]
%%%             results_iter    The iteration information of 'results'
%%%             P               The learned projection matrix
%% Version
%%%     Implementation          2022-05-28
    options=defaultOptions(options,...
                'T',10,...              %%% The total iteration times
                'dim',100,...           %%% The dimension reduced
                'k',10,...              %%% The number of KNN
                't',10,...              %%% The iteration times
                'alpha',0.1,...         %%% The relative weight of Sb, i.e., Sw-alpha*Sb
                'beta',1,...            %%% The weight of nuclear norm w.r.t J (Z)
                'theta',1,...           %%% The weight of L2,1 norm w.r.t E
                'lambda',0.1,...        %%% The weight of scatters w.r.t P
                'mu',0.1,...           %%% The lagrange coefficient
                'muMax',1e3,...         %%% The maximum value of `mu`
                'rho',1.01);             %%% The increase rate of `mu`
    %% parameters
    T=options.T;
    k=options.k;
    myeps=1e-4;
    %% Initialization
    results_iter=[];
    [m,n]=size(trainX);
    % Init G & V
    G=1/n*withinScatter(trainX,trainY);
    V=1/n*betweenScatter(trainX,trainY);
    dist=EuDist2(trainX',trainX');
    [S,rr]=similarMatrix_CAN(dist,k,-1);
    [P,~]=eigs(options.lambda*G-options.lambda*options.alpha*V+myeps*eye(m),eye(m),options.dim,'sm');
    for i=1:T
        % Solve Z and E by Algorithm 1
        [Z,~] = LRDAGP_solveAlg1(trainX,P,options);
        % Solev P by Algorithm 2
        S=(S+S')/2;
        D=diag(sparse(sum(S)));
        L=D-S;
        L=L./norm(L,'fro');
        XLX=trainX*L*trainX';
        P = LRDAGP_solveAlg2(trainX,P,Z,XLX,G,V,options);
        % Update S by CAN
        Zs=real(P'*trainX);
        dist=EuDist2(Zs');
        S=similarMatrix_CAN(dist,k,rr);
        % Classification
        Zt=real(P'*testX);
        Ytpseudo=classifyKNN(Zs,trainY,Zt,1);
        results=MyClusteringMeasure(testY,Ytpseudo);%[ACC MIhat Purity]';
        for index=1:3
            results_iter(index,i)=results(index);
        end
        fprintf('[%d]-th acc:%.4f, MIhat: %.4f, Purity:%.4f\n',i,...
            results(1),results(2),results(3));
        
    end
end