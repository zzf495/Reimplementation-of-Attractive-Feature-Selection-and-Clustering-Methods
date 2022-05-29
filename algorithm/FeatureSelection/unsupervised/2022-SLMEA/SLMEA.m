function [results,results_iter,W]=SLMEA(X,Y,options)
%% Implementation of SLMEA
%%% Authors:                    Shang et al.
%%% Titl:                       2022-Sparse and low-dimensional representation with maximum entropy adaptive graph for feature selection
%% intput:
%%% X:                     The samples, m*n
%%% Y:                     The labels of samples, n*1
%% options
%%% T                            The iterations of V
%%% dim                          The dimension selected
%%% alpha                        The weight of NMF w.r.t features |X'-X'WH|
%%% beta                         The weight of L(2,1/2)-(1/2) norm
%%% gamma                        The weight of manfiold regularization
%%% lambda                       The weight of entropy
%% output:
%%% results                      The results (list) [acc,NMI,purity]
%%% results_iter                 The iteration information of 'results'
%%% W                            The learned feature selection matrix
%% Version
%%%     Implementation           2022-05-19
    options=defaultOptions(options,...
                'T',10,...      %% The iterations
                'dim',80,...    %% The dimension selected
                'alpha',1e-3,...   %% The weight of NMF w.r.t features |X'-X'WH|
                'beta',0.01,...  %% The weight of L(2,1/2)-(1/2) norm
                'gamma',100,...%% The weight of manfiold regularization
                'lambda',1);    %% The weight of entropy
    %% parameters
    T=options.T;
    dim=options.dim;
    alpha=options.alpha;
    beta=options.beta;
    gamma=options.gamma;
    lambda=options.lambda;
    %% Initialization
    eta=0.1;
    X=normr(X')';
    results_iter=[];
    myeps=1e-8;
    C=length(unique(Y));
    [m,n]=size(X);
    XX=X*X';
    % Init H
    [~,H]=litekmeans(X',C); % C*m
    [~,F]=litekmeans(X,C);F=F'; % n*C
    W=(H'*H+eta*eye(m))\H'; % m*C
    W=max(W,myeps);
    Gn=centeringMatrix(n);
    for i=1:T
        % Update S^H by Eq.(38)
        dist=EuDist2(H',H',0)+myeps;
        expDist=exp(dist/(2*lambda));
        sumExp=sum(expDist,1);
        SH=expDist./sumExp;
        % Update S^F by Eq.(40)
        dist=EuDist2(F,F,0);
        expDist=exp(dist/(2*lambda));
        sumExp=sum(expDist,1);
        SF=expDist./sumExp;
        % Update H by Eq.(24)
        DH=diag(sparse(sum(SH)));
        left=alpha*W'*XX+gamma*H*SH;
        right=alpha*W'*XX*W*H+gamma*H*DH;
        res=left./right;
        H=IterativeMultiplicativeUpdate(H,res);
        % Update F by GPI
        DF=diag(sparse(sum(SF)));
        LF=DF-SF;
        A1=Gn+gamma*LF;% support matrix
        A2=Gn*X'*W;% support matrix
        %% Method 2
        try
            R = GPI(A1,A2,[]);
            [Ur,~,Vr]=mySVD(R);
            F=Ur*Vr';
        catch ME
            warning('An error occurs when runing GPI'); 
            break;
        end

        % Update W by Eq.(21) 
        %%% compute U
        U=(4*diag(1./sum(power(W.*W+myeps,3/2),2)));
        %%% compute W
        left=alpha*XX*H'+X*Gn*F;
        right=alpha*XX*W*(H*H')+beta*U*W+X*Gn*X'*W;
        res=left./right;
        W=IterativeMultiplicativeUpdate(W,res);
        % Scores
         [~,results] = getFeatureSelectionResults(X,Y,W,dim,C);
        for index=1:3
            results_iter(index,i)=results(index);
        end
        fprintf('[%d]-th acc:%.4f, MIhat: %.4f, Purity:%.4f\n',i,...
            results(1),results(2),results(3));
    end
end