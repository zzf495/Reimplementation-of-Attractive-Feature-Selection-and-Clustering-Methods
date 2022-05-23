function result=IterativeMultiplicativeUpdate(X,gradient)
%% input:
%%%     X :         The matrix waiting for update (m*n)
%%%     gradient:   The gradient of X (m*n)
%% Output:
%%%     result:     The matrix updated (m*n)
    myeps=1e-8;
    gradient=gradient+((abs(gradient)<myeps).*myeps);
    result=X.*gradient;
end

