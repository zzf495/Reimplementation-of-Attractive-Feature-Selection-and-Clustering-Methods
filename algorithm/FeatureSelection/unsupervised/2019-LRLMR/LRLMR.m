function [results,results_iter,W]=LRLMR(X,Y,options)
%% Implementation of LRLMR
%%% Authors:                    Tang et al.
%%% Titl:                       2019-Unsupervised feature selection via latent representation learning and manifold regularization
%% intput:
%%% X:                     The samples, m*n
%%% Y:                     The labels of samples, n*1
%% options
%%% T                            The iterations of V
%%% t                            The iterations of W
%%% dim                          The dimension selected
%%% alpha                        The weight of L2,1 norm
%%% beta                         The weight of A-VV'
%%% gamma                        The weight of manifold regularization
%%% k                            The KNN number
%% output:
%%% results                      The results (list) [acc,acc2,NMI,purity]
%%% results_iter                 The iteration information of 'results'
%%% W                            The learned feature selection matrix
%% Version
%%%     Implementation           2022-05-19
    options=defaultOptions(options,...
                'T',10,...
                't',10,...
                'dim',80,...
                'alpha',1,... 
                'beta',1e-4,... 
                'gamma',1e-4,...
                'k',10);
    %% parameters
    T=options.T;
    t=options.t;
    dim=options.dim;
    alpha=options.alpha;
    beta=options.beta;
    gamma=options.gamma;
    k=options.k;
    %% Initialization
    results_iter=[];
    myeps=1e-8;
    C=length(unique(Y));
    [m,n]=size(X);
    XX=X*X';
    % Init L by Eq.(6)
    clear manifold;
    manifold.k = k;
    manifold.Metric = 'Euclidean';
    manifold.WeightMode = 'HeatKernel';
    manifold.NeighborMode = 'KNN';
    L=computeL(X,manifold);
    L=L./norm(L,'fro');
    XLX=X*L*X';
    % Init A by Eq.(6)
    clear manifold;
    manifold.k = 0;
    manifold.Metric = 'Euclidean';
    manifold.WeightMode = 'HeatKernel';
    manifold.NeighborMode = 'KNN';
    A=lapgraph(X',manifold);
    % Init V by random
    V=rand(n,C);
    % Init G (Lambda in paper)
    G=eye(m);
    for i=1:T
        XV=X*V;
        for j=1:t
           % Update W by Eq.(11)
           W=(XX+alpha*G+gamma*XLX)\(XV);
           % Update G (Lambda) by Eq.(8)
           G=updateL21(W);
        end
        % Update V
        left=(X'*W)+2*beta*(A*V);
        right=V+2*beta*(V*V')*V;
        res=left./right;
        V=IterativeMultiplicativeUpdate(V,res);
        % scores
        [~,results] = getFeatureSelectionResults(X,Y,W,dim,C);
        for index=1:3
            results_iter(index,i)=results(index);
        end
        fprintf('[%d]-th acc:%.4f, MIhat: %.4f, Purity:%.4f\n',i,...
            results(1),results(2),results(3));
    end
end