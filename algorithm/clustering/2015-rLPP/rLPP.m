function [results,results_iter,W]=rLPP(X,Y,options)
%% Implementation of rLPP
%%% Authors                      Wang et al.
%%% Titl                         2015-Learning Robust Locality Preserving Projection via p-Order Minimization
%% intput:
%%% X                            The samples, m*n
%%% Y                            The labels of samples, n*1
%% options
%%% T                            The iterations
%%% dim                          The dimensions
%%% Metric                       The `Metric` of `lapgraph`
%%%%                         -   0   
%%%%                                Use `Euclidean`
%%%%                         -   1   
%%%%                                Use `Cosine`
%%% WeightMode                   The `WeightMode` of `lapgraph`
%%%%                         -   0   
%%%%                                Use `HeatKernel`
%%%%                         -   1   
%%%%                                Use `Cosine`
%%% k                            The number of KNN
%%% p                            The p-norm (0<p<=2)
%%% lambda                       The weight of regularization
%% output:
%%% results                      The results (list) [acc,NMI,purity]
%%% results_iter                 The iteration information of 'results'
%%% W                            The learned feature selection matrix
%% Version
%%%     Implementation          2022-06-03
    options=defaultOptions(options,...
                'T',10,...          %% The iterations
                'dim',100,...       %% The dimensions
                'Metric',0,...      %% The `Metric` of `lapgraph`
                'WeightMode',0,...  %% The `WeightMode` of `lapgraph`
                'k',10,...          %% The number of KNN
                'p',2,....          %% The p-norm (0<p<=2)
                'lambda',1);        %% The weight of regularization
     
    %% parameters
    T=options.T;
    dim=options.dim;
    p=options.p;
    lambda=options.lambda;
    [m,n]=size(X);
    %% Init lapgraph S
     opt=struct();
     switch options.Metric
         case 0
             opt.Metric='Euclidean';
         case 1
             opt.Metric='Cosine';
     end
     switch options.WeightMode
         case 0
             opt.WeightMode='HeatKernel';
         case 1
             opt.WeightMode='Cosine';
     end
     opt.k=options.k;
     opt.NeighborMode='KNN';
     [S, ~] = lapgraph(X',opt);
     D=diag(sparse(sum(S)));
     % Init W
     XDX=X*D*X';
     [W,~]=eigs(XDX,eye(m),dim,'lm');
     WX=W'*X;
    for i=1:T
        % Update S' by Eq.(4)
        dist=power(EuDist2(WX'),p-2);
        dist(dist==inf)=0;
        dist(isnan(dist))=0;
        Sprime=p/2*(S.*dist);
        Dprime=diag(sparse(sum(Sprime)));
        % Update Lprime
        Lprime=Dprime-Sprime;
        XLprimeX=X*Lprime*X';
        % Update W
        [W,~]=eigs(XLprimeX+lambda*eye(m),XDX,dim,'sm');
        % Scores
        [WX,results] = getClusteringResults(X,Y,W);
        for index=1:3
            results_iter(index,i)=results(index);
        end
        fprintf('[%d]-th acc:%.4f, MIhat: %.4f, Purity:%.4f\n',i,...
            results(1),results(2),results(3));
    end
end

