function [results,results_iter,W] = DSLRL(X,Y,options)
%% Implementation of DSLRL
%%% Authors:                    Shang et al.
%%% Titl:                       2021-Dual space latent representation learning for unsupervised feature selection
%% intput:
%%% X:                          The samples, m*n
%%% Y:                          The labels of samples, n*1
%% options
%%% T                            The iterations of V
%%% t                            The iterations of W
%%% d                            The dimension reduced
%%% alpha                        The weight of L2-1 for W
%%% beta                         The weight of A-VV'
%%% gamma                        The weight of manfiold regularization
%%% lambda                       The weight of WW'=I
%% output:
%%% results                      The results (list)
%%% results_iter                 The iteration information of 'results'
%%% W                            The learned feature selection matrix
%% Version
%%%     Implementation          2022-05-19
    options=defaultOptions(options,...
                'T',10,...      %% The iterations
                'dim',80,...    %% The dimension reduced
                'alpha',1e3,...   %% The weight of L2-1 for W
                'beta',1e2,...  %% The weight of A-VV'
                'gamma',1e-3,...%% The weight of manfiold regularization
                'lambda',1e-3);    %% The weight of WW'=I
    %% parameters
    T=options.T;
    dim=options.dim;
    alpha=options.alpha;
    beta=options.beta;
    gamma=options.gamma;
    lambda=options.lambda;
    %% Initialization
    X=normr(X')';
%     X=L2Norm(X')';
    results_iter=[];
    C=length(unique(Y));
    [m,n]=size(X);
    eta=0.1;
    % Init A by Eq.(7)
    clear manifold;
    manifold.k = 0;
    manifold.Metric = 'Euclidean';
    manifold.WeightMode = 'HeatKernel';
    manifold.NeighborMode = 'KNN';
    A=lapgraph(X',manifold);
    % Init B by Eq.(8)
    clear manifold;
    manifold.k = 0;
    manifold.Metric = 'Euclidean';
    manifold.WeightMode = 'HeatKernel';
    manifold.NeighborMode = 'KNN';
    B=lapgraph(X,manifold);
    % Init V
    [~,V]=litekmeans(X,C);V=V'; % n*C
    % Init H
    H=eye(m);
    % Init W
    W=(X*X'+eta*eye(m))\(X*V);
    W=max(W,1e-8);
    X=X';% input X: n*m
    for i=1:T
       	% Update W by Eq.(20)
        left=X'*V+2*beta*B*W+2*lambda*W;
        right=X'*X*W+alpha*H*W+2*(gamma+lambda)*W*W'*W;
        res=left./right;
        W=IterativeMultiplicativeUpdate(W,res);
        % Update H by Eq.(15)
        H=updateL21(W);
        % Update V by Eq.(23)
        left=X*W+2*beta*A*V;
        right=V+2*beta*V*V'*V;
        res=left./right;
        V=IterativeMultiplicativeUpdate(V,res);
        % scores
        [~,results] = getFeatureSelectionResults(X',Y,W,dim,C);
        for index=1:3
            results_iter(index,i)=results(index);
        end
        fprintf('[%d]-th acc:%.4f, MIhat: %.4f, Purity:%.4f\n',i,...
            results(1),results(2),results(3));
    end
end