function [results,results_iter,W] = DSDPL(trainX,trainY,testX,testY,options)
%% Notice
%%%     The codes implemented require to cope with the data by 'zscore' or
%%%     'normr' fist (so as to avoid the calculation of large values).
%% Implementation of DSDPL
%%% Authors:                    Belous et al.
%%% Titl:                       2021-Dual subspace discriminative projection learning
%% intput:
%%% trainX:                     The traing samples, m*n1
%%% trainY:                     The labels of training samples, n1*1
%%% testX:                      The test samples, m*n2
%%% testY:                      The labels of test samples, n2*1
%% options
%%% T                            The iterations
%%% dim                          The dimension reduced
%%% mu                           The weight of between-class scatter
%%% lambda1                      The weight of L2,1 norm w.r.t Q and R
%%% lambda2                      The weight of L1 norm w.r.t. E0 and E1
%%% beta                         The initial value of Lagrangian coefficient
%%% betaMax                      The maximum Lagrangian coefficient
%%% rho                          The increase rate of Lagrangian coefficient
%% output:
%%% results                      The results (list) [acc,NMI,purity]
%%% results_iter                 The iteration information of 'results'
%%% W                            The learned projection matrix
%% Version
%%%     Implementation          2022-05-22
    options=defaultOptions(options,'T',10,...
                           'mu',1e3,...           %% The weight of between-class scatter
                           'lambda1',0.01,...    %% The weight of L2,1 norm w.r.t Q and R (0.01)
                           'lambda2',0.001,...  %% The weight of L1 norm w.r.t. E0 and E1
                           'beta',0.1,...       %% The initial value of Lagrangian coefficient
                           'beta_Max',1e2,...   %% The maximum Lagrangian coefficient
                           'rho',1.01,...       %% The increase rate of Lagrangian coefficient
                           'd',30);             %% The dimension reduced
    %% Parameters Setting
    T=options.T;
    beta=options.beta;
    lambda1=options.lambda1;
    lambda2=options.lambda2;
    mu=options.mu;
    d=options.d;
    beta_Max=options.beta_Max;
    rho=options.rho;
    %% Data process
    [m,n]=size(trainX);
    C=length(unique(trainY));
    %% initialization
    Sw=1/n*withinScatter(trainX,trainY);
    Sb=1/n*betweenScatter(trainX,trainY);
    V={};
    R={};
    E={};
    W={};
    E0=zeros(m,n);
    Ck={};
    Ck0=0;
    % init Xc,Yc
    Xc={};
    Yc={};
    
    for k=1:C
        tmpXc=trainX(:,trainY==k);
        tmpYc=hotmatrix(trainY(trainY==k),C);
        Xc{k}=tmpXc;
        [m_Xc,d_Xc]=size(Xc{k});
        Yc{k}=tmpYc';
        V{k}=rand(m_Xc,d_Xc); %% If V=0, a error occurs
        R{k}=zeros(m_Xc,d_Xc);
        E{k}=zeros(m_Xc,d_Xc);
        W{k}=zeros(C,d_Xc);
        Ck{k}=zeros(m_Xc,d_Xc);
    end
    Q=rand(m,d);
    % loop T
    D=eye(m,m);
    for i=1:T
       % Step 1, Update P
       M0=(trainX-E0+Ck0/beta);
       Mi={};
       XQMi=0;
       for k=1:C % m*1 - m*d*d*m*m*1 - m*1
           Mi{k}=(Xc{k}-V{k}*R{k}'*Xc{k}-E{k}+Ck{k}/beta); %
           XQMi=XQMi+Mi{k}*Xc{k}'*Q;
       end
       [U1,~,V1]=mySVD(M0*trainX'*Q+XQMi,d);
       P=U1*V1'; % m*d
       % Step 2, Update Q
       XMPi=0;
       sumSqureXc=0;
       for k=1:C % m*1 - m*d*d*m*m*1 - m*1
           XMPi=XMPi+Xc{k}*Mi{k}'*P;
           sumSqureXc=sumSqureXc+Xc{k}*Xc{k}';
       end
       
       Q=(lambda1*D+beta*(trainX*trainX'+ sumSqureXc))\(beta*(trainX*M0'*P)+XMPi); %m*d
       D=updateL21(Q);
       % Step 3/4,
       for k=1:C
          % Update Ri
          if i==1
              F=eye(m,m);
          else
              F=updateL21(R{k});
          end
          tmpM=Xc{k}-P*Q'*Xc{k}-E{k}+Ck{k}/beta;
          R{k}= (lambda1*F+2*(Sw-mu*Sb)+(2+beta)*Xc{k}*Xc{k}')\(beta*Xc{k}*tmpM'*V{k}+2*Xc{k}*Yc{k}'*W{k});
          % Update Vi
          [U2,~,V2]=mySVD(tmpM*Xc{k}'*R{k},d);
          V{k}=U2*V2';
          % Update Wi
          [U3,~,V3]=mySVD(Yc{k}*Xc{k}'*R{k},d);
          W{k}=U3*V3';
          % Update Ei
          E{k}=Xc{k}-P*Q'*Xc{k}-V{k}*R{k}'*Xc{k}-Ck{k}/beta;
          E{k}=shrink(E{k},lambda2/beta);
          Ck{k}=Ck{k}+beta*(Xc{k}-P*Q'*Xc{k}-V{k}*R{k}'*Xc{k}-E{k});
       end
       % Update E0
       E0=trainX-P*Q'*trainX+Ck0/beta;
       E0=shrink(E0,lambda2/beta);
       % Update lagrangian multipliers C
       Ck0=Ck0+beta*(trainX-P*Q'*trainX-E0);
       % Update beta;
       beta=min(rho*beta,beta_Max);
       % Classification
       prob=zeros(C,size(testX,2));
       for k=1:C
          tmp=(W{k}*R{k}'*testX);
          prob(k,:)=tmp(k,:);
       end
       [~,Ytpseudo]=max(prob,[],1);
       results=MyClusteringMeasure(testY,Ytpseudo);%[ACC ACC2 MIhat Purity]';
       for index=1:3
           results_iter(index,i)=results(index);
       end
       fprintf('[%d]-th acc:%.4f, MIhat: %.4f, Purity:%.4f\n',i,...
            results(1),results(2),results(3));
    end
    
end

