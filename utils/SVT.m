function res = SVT(W,tau)
%% paper: A singular value thresholding algorithm for matrix completion
%% input:
%%%     W: dealed matrix
%%%     tau: threshold
%% ouput:
%%%     res:SVT results
    [U,S,V]=svd(W,'econ');
    %% equal to S=max(0,S-tau)+min(0,S+tau);
    S=sign(S).*max(abs(S)-tau,0);
    res=U*S*V';
end

