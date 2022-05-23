function [W] = GPI(A,B,options)
%% Solve:
%%%         min tr(W'AW-2W'B)       s.t. W'W=I
%%%      => max tr(W' hatA W+2W'B)  s.t. W'W=I, hatA=alpha*I-A
%% input:
%%% W       learned matrix
%%% A       W'AW  (m*m)
%%% B       2 W'B  (m*k)
%% options:
%%%         T: iteration (default:1e3)
%%%         maxIter:maxRandom (default:1e3)
%%%         precision: the convergence precision (default:1e-4)
if  nargin==2
    options=struct();
end
if ~isfield(options,'T')
    options.T=1e3;
end
if ~isfield(options,'precision')
   options.precision=1e-4; 
end
T=options.T;
precision=options.precision;
n=size(A,1);
%% mu : the largest eigenvalue of A
[U,V] = eig(A);
[~, index] = sort(diag(V),'ascend');
mu=diag(V);mu=max(mu);
Atau=mu*eye(n)-A;
W = U(:, index(1:size(B,2)));
% try chol(Atau);
% %     fprintf('Matrix is symmetric positive definite.\n');
% catch ME
%     warning('[GPI] Atau=mu*eye(n)-A is not a positive definite matrix!');
% end
lastVal=Inf;
for i=1:T
    beforeW=W;
    M=2*Atau*W+2*B;
    [U,~,V]=svd(M,'econ');
    W=U*V';
    val=norm(beforeW-W,'inf');
    if abs(lastVal-val)<precision &&i>=3
        break;
    else
        lastVal=val;
    end
    if i==T
       warning('[GPI] No convergence (iteration > maximum T).'); 
    end
end
end

