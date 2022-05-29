function [results,results_iter,Z] = DLUFS(X,Y,options)
%% Notice
%%% Official codes (python implementation) are available at https://github.com/mohsengh/DLUFS/
%% Data process: zscore
%% Implementation of DLUFS
%%% Authors:                     Ghassemi Parsa et al.
%%% Titl:                       2022-Low-rank dictionary learning for unsupervised feature selection
%% intput:
%%% X:                     The samples, m*n
%%% Y:                     The labels of samples, n*1
%% options
%%% T                            The iterations
%%% dim                       	 The dimension selected
%%% alpha                        The weight of manifold regularization
%%% sigma                        The weight of Gaussian kernel
%%% k                            The number of KNN
%%% lambda                       The weight of L2,1-norm of Z
%% output:
%%% results                      The results (list)
%%% results_iter                 The iteration information of 'results'
%%% Z                            The learned feature selection matrix
%% Version
%%%     Implementation          2022-05-23
    %% Parameter setting
    options=defaultOptions(options,...
                'T',10,...    %% The iterations
                'dim',300,... %% The dimension selected
                'alpha',1e-3,... %% The weight of manifold regularization
                'sigma',1e2,... %% The weight of Gaussian kernel
                'k',10,... %% The number of KNN
                'lambda',1e-3); %% The weight of L2,1-norm of Z
    %% Parameter Setting
    T=options.T;
    dim=options.dim;
    alpha=options.alpha;
    lambda=options.lambda;
    sigma=options.sigma;
    k=options.k;
    eta=1e-16;
    %% Initialization
    C=length(unique(Y));
    m=size(X,1);
    results_iter=[];
    % Compute L
    manifold.NeighborMode = 'KNN';
    manifold.k = k;
    manifold.t =sigma;
    manifold.WeightMode = 'Heatkernel';
    manifold.Metric='Euclidean';
    L=computeL(X,manifold);
    XX=X'*X;
    Z=X;
    for i=1:T
        % UPdate B by Eq.(16)
        Sw=(Z*Z'+eta*eye(m));
        Sb=Z*XX*Z';
        res=Sw\Sb;
        [Ub,~,Vb]=mySVD(res,dim);
        B=Ub*Vb';
        % Update A by Eq.(10)
        A=(X*Z'*B')/(B*Sw*B'+eta*eye(m));
        % Update D by Eq.(19)
        D=updateL21(Z);
        % Update Z by Eq.(21)
        AB=A*B;
        E=AB'*AB+lambda*D;
        F=alpha*L;
        G=AB'*X;
        sylvester(E,F,G);
        % Scores
         [~,results] = getFeatureSelectionResults(X,Y,Z,dim,C);
        for index=1:3
            results_iter(index,i)=results(index);
        end
        fprintf('[%d]-th acc:%.4f, MIhat: %.4f, Purity:%.4f\n',i,...
            results(1),results(2),results(3));
    end
end

