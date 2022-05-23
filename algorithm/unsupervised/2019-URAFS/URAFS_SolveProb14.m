function [W] = URAFS_SolveProb14(X,H,S,F,lambda,T)
    [m,~]=size(X);
    % Initialize D by eye
    D=eye(m,m);
    for i=1:T
        % Compute Q,B by Eq.(20)
        temp=S+lambda*D;
        [Ut,Sigma,Vt]=mySVD(temp);
        Sigma(Sigma<0)=0;
        squreSigma=Sigma.^0.5;
        inverseSqrtSigma=diag(1./(diag(squreSigma)));
        inverseSqrtSigma(isinf(inverseSqrtSigma))=0;
        SlambdaD=Ut*(inverseSqrtSigma)*Vt';
        B= (SlambdaD)*X*H*F;
        % Update Q
        [Ub,~,Vb]=mySVD(B);
        Q=Ub*Vb';
        % Update W
%         W=S+lambda*D;
        W=SlambdaD*Q;
        % Update D
        D=updateL21(W);
    end
end

