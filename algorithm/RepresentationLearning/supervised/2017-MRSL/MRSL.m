function [results,results_iter,W]=MRSL(trainX,trainY,testX,testY,options)
%% Oficial codes: https://github.com/DarrenZZhang/MSRL
%% Implementation of MRSL (Semi-supervised version)
%%% Authors:        Zhang et al.
%%% Titl:           2017-Marginal Representation Learning With Graph Structure Self-Adaptation
%% intput:
%%% trainX:         The traing samples, m*n1
%%% trainY:         The labels of training samples, n1*1
%%% testX:          The test samples, m*n2
%%% testY:          The labels of test samples, n2*1
%% options
%%% T:              The iterations
%%% s:              The latent dimension of projection matrix
%%% mu:             The regularization used for initialize projection
%%%                 matrix W
%%% lambda:         The weight of manifold regularization
%%% beta:           The weight of norm \|W\|_F^2
%%% gamma:          The weight of reconstruction 
%% output:
%%% results                      The results (list) [acc,acc2,NMI,purity]
%%% results_iter                 The iteration information of 'results'
%%% W                            The learned projection matrix
%% Version
%%%     Implementation          2022-05-18
    options=defaultOptions(options,...
                's',10,...          %%% The latent dimension of projection matrix
                'T',10,...          %%% The iterations
                'mu',1e-4,...       %%% The weight of inter-class term in LDA (1e-4 in paper)
                'lambda',1,...      %%% The weight of manifold regularization
                'beta',1,...        %%% The weight of norm \|W\|_F^2
                'gamma',1,...       %%% The weight of reconstruction 
                'k',10);     %%% The KNN numbers
    %% parameters
    T=options.T;
    mu=options.mu;
    beta=options.beta;
    lambda=options.lambda;
    gamma=options.gamma;
    k=options.k;
    s=options.s;
    %% Initialization
    results_iter=[];
    trainX=normr(trainX')';
    testX=normr(testX')';
    
    C=length(unique(trainY));
    [m,~]=size(trainX);
    n2=size(testX,2);
    X=[trainX,testX];
    %% Set W
    hotY1=hotmatrix(trainY,C);
    W=((trainX*trainX')+mu*eye(m))\(trainX*hotY1);
    %% Set P
    distX = EuDist2(X',X');
    [P,rr]=similarMatrix_CAN(distX,k,-1);
    A=eye(m,s);
    R=[hotY1;eye(n2,C)]';
    for i=1:T
        % Update B by Eq.(19)
        B=A'*W;
        % Update W by Eq.(21)
        %%% compute L
        Dp=diag(sparse(sum(P)));
        L=Dp-P;
        L=L./norm(L,'fro');
        %%% compute W
        G=X*X'+lambda*X*L*X'+(beta+gamma)*eye(m);
        W=(G-gamma*(A*A'))\(X*R');
        % Update A by Eq.(22)
        [Ua,~,Va]=mySVD(W*B',s);
        A=Ua*Va';
        % Classification
        [~,Ytpseudo]=max(W'*testX,[],1);Ytpseudo=Ytpseudo';
        YY=[trainY;Ytpseudo];
        % Update R by Algorithm 1
        F=W'*X;
        [R] = MRSL_getR(F,YY);
        % Update P by Eq.(36)
        distF=   EuDist2(F',F');
        [P,~]=similarMatrix_CAN(distF,k,rr);
        results=MyClusteringMeasure(testY,Ytpseudo);%[ACC ACC2 MIhat Purity]';
        for index=1:3
            results_iter(index,i)=results(index);
        end
        fprintf('[%d]-th acc:%.4f, MIhat: %.4f, Purity:%.4f\n',i,...
            results(1),results(2),results(3));
    end
end

