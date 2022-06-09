function [WX,resultsFinal] = getClusteringResults(X,Y,W,C,options)
%% Formula
%%%     select top-k highest scores of features in W'X
%% Input:
%%%     X              The feature, m*n
%%%     Y              The labels, n*1
%%%     W              The feature selection matrix, m*d
%%%     C              The clustering number (default `length(unique(Y))`)
%% Output:
%%%     WX             The projection subspace, d*n
%%%     results        The clustering results [acc,NMI,purity]
    if nargin<=4
        options=struct();
    end
    if nargin<=3
       C=length(unique(Y)); 
    end
    options=defaultOptions(options,...
                'T',10,...     %% The repeat times of kmeans
                'MaxIter',100,... %% Options of 'litekmeans'
                'Replicates',10,...%% Options of 'litekmeans'
                'supervisedFlag',0); %% Options of 'MyClusteringMeasure'
   	WX=W'*X;
    resultsAll=[];
    for i=1:options.T
        Ypseudo=litekmeans(WX',C,'MaxIter',options.MaxIter,'Replicates',options.Replicates);
        results=MyClusteringMeasure(Y,Ypseudo);%[ACC ACC2 MIhat Purity]';
        resultsAll=[resultsAll,results];
    end
    resultsFinal=mean(resultsAll,2);
end

