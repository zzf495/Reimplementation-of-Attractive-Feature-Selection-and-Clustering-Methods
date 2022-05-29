function [results,results_iter,W]=AGUFS(X,Y,options)
%% Implementation of AGUFS
%%% Authors:                    Huang et al.
%%% Titl:                       2021-Adaptive graph-based generalized regression model for unsupervised feature selection
%% intput:
%%% X:                     The samples, m*n
%%% Y:                     The labels of samples, n*1
%% options
%%% T                            The iterations
%%% t                            The iterations of GPI and Algorithm 1
%%% dim                          The dimension selected
%%% alpha                        The weight of manifold regularization
%%% k                            The number of KNN
%%% lambda                       The weight of L2,1-norm w.r.t W
%% output:
%%% results                      The results (list) [acc,acc2,NMI,purity]
%%% results_iter                 The iteration information of 'results'
%%% W                            The learned feature selection matrix
%% Version
%%%     Implementation          2022-05-23
    options=defaultOptions(options,...
                'T',10,...        %% The iterations
                't',10,...        %% The iterations of GPI and Algorithm 1
                'dim',60,...      %% The dimension selected
                'alpha',1e3,...   %% The weight of manifold regularization
                'k',10,...        %% The number of KNN
                'lambda',1e3);    %% The weight of L2,1-norm w.r.t W
    %% parameters
    T=options.T;
    t=options.t;
    dim=options.dim;
    alpha=options.alpha;
    lambda=options.lambda;
    k=options.k;
    %% Initialization
    results_iter=[];
    C=length(unique(Y));
    [~,n]=size(X);
    % Init F
    [~,F]=litekmeans(X,C);F=F'; % n*C
    [Uf,~,~]=mySVD(F);
    F=Uf; % F'F=I
    % Init H
    H=centeringMatrix(n);
    % Init St
    XHX=X*H*X';
    % Init S by Eq.(22)
    distX = EuDist2(X',X');
    [S,rr]=similarMatrix_CAN(distX,k,-1);
    for i=1:T
        % compute L
        DS=diag(sparse(sum(S)));
        L=DS-S;
        XLX=X*L*X';
        % Update W by Algorithm 1
        W=AGUFS_SolveProb14(X,H,XHX,XLX,F,alpha,lambda,t);
        % Update L by (7)
        P=diag(sparse(sum(S)));
        Ls=P-S;
        A=H+0.5*alpha*Ls;
        CA=H*X'*W;
        % Update F by Algorithm 2 (GPI)
        opt.T=t;
        F=GPI(A,CA,opt);
        % Update S by Eq.(22)
        distX = EuDist2(F,F);
        [S,~]=similarMatrix_CAN(distX,k,rr);
        S=real(S);
        %% Classification
        % Select top d ranked features (descending order) as the results
        % scores
        [~,results] = getFeatureSelectionResults(X,Y,W,dim,C);
        for index=1:3
            results_iter(index,i)=results(index);
        end
        fprintf('[%d]-th acc:%.4f, MIhat: %.4f, Purity:%.4f\n',i,...
            results(1),results(2),results(3));
    end
end