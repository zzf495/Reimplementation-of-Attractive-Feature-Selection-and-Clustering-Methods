function [W] = AGUFS_SolveProb14(X,H,XHX,XLX,F,alpha,lambda,T)
    [m,~]=size(X);
    % Initialize D by eye
    D=eye(m,m);
    for i=1:T
        % Compute Q,B by Eq.(20)
        temp=XHX+alpha*XLX+lambda*D;
        [Ut,Sigma,Vt]=mySVD(temp);
        Sigma(Sigma<0)=0;
        squreSigma=Sigma.^0.5;
        inverseSqrtSigma=diag(1./(diag(squreSigma)));
        inverseSqrtSigma(isinf(inverseSqrtSigma))=0;
        SlambdaD=Ut*(inverseSqrtSigma)*Vt';
        B= (SlambdaD)*X*H*F;
        % Update A
        [Ub,~,Vb]=mySVD(B);
        A=Ub*Vb';
        % Update W
        W=SlambdaD*A;
        % Update D
        D=updateL21(W);
    end
end

