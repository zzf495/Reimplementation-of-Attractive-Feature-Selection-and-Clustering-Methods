function [X_new,resultsFinal] = getFeatureSelectionResults(X,Y,W,dim,C,options)
%% Formula
%%%     select top-k highest scores of features in X'W
%% Input:
%%%     X              The feature, m*n
%%%     Y              The labels, n*1
%%%     W              The feature selection matrix, m*m
%%%     dim            The dimension reduced
%% Output:
%%%     newX           The selected feautre sample, k*n
%%%     results        The clustering results [acc,NMI,purity]
    if nargin<=5
        options=struct();
    end
    if nargin<=4
       C=length(unique(Y)); 
    end
    options=defaultOptions(options,...
                'T',10,...     %% The repeat times of kmeans
                'MaxIter',100,... %% Options of 'litekmeans'
                'Replicates',10,...%% Options of 'litekmeans'
                'supervisedFlag',0); %% Options of 'MyClusteringMeasure'
    score=sum((W.*W),2);
    [~,index]=sort(score,'descend');
    X_new = X(index(1:dim),:);
    resultsAll=[];
    for i=1:options.T
        Ypseudo=litekmeans(X_new',C,'MaxIter',options.MaxIter,'Replicates',options.Replicates);
        results=MyClusteringMeasure(Y,Ypseudo,options.supervisedFlag);%[ACC ACC2 MIhat Purity]';
        resultsAll=[resultsAll,results];
    end
    resultsFinal=mean(resultsAll,2);
end

