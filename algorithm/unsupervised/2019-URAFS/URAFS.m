function [results,results_iter,W]=URAFS(X,Y,options)
%% Implementation of URAFS
%%% Authors:                    Li et al.
%%% Titl:                       2019-Generalized Uncorrelated Regression with Adaptive Graph for Unsupervised Feature Selection
%% intput:
%%% X:                     The samples, m*n
%%% Y:                     The labels of samples, n*1
%% options
%%% T                            The iterations of V
%%% t                            The iterations of GPI and Algorithm 1
%%% dim                          The dimension reduced
%%% alpha                        The weight of manfiold regularization 
%%% beta                         The weight of Gaussian kernel (S)
%%% lambda                       The weight of L2,1 of W
%% output:
%%% results                      The results (list) [acc,acc2,NMI,purity]
%%% results_iter                 The iteration information of 'results'
%%% W                            The learned feature selection matrix
%% Version
%%%     Implementation          2022-05-19
    options=defaultOptions(options,...
                'T',10,...        %% The iterations
                't',10,...        %% The iterations of GPI and Algorithm 1
                'dim',60,...      %% The dimension reduced
                'alpha',1e3,...   %% The weight of NMF w.r.t features |X'-X'WH|
                'beta',1e3,...    %% The weight of Gaussian kernel (S)
                'lambda',1e3);    %% The weight of entropy
    %% parameters
    T=options.T;
    t=options.t;
    dim=options.dim;
    alpha=options.alpha;
    beta=options.beta;
    lambda=options.lambda;
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
    St=X*H*X';
    for i=1:T
        % Update S by Eq.(31)
        dist=EuDist2(F,F,0);
        expDist=exp(-dist/(2*beta));
        sumExp=sum(expDist,1);
        S=expDist./sumExp;
        S=(S+S')/2;
        % Update W by Algorithm 1
        W=URAFS_SolveProb14(X,H,St,F,lambda,t);
        % Update L by (7)
        P=diag(sparse(sum(S)));
        Ls=P-S;
        A=H+2*alpha*Ls;
        CA=H*X'*W;
        % Update F by Algorithm 2 (GPI)
        opt.T=t;
        F=GPI(A,CA,opt);
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