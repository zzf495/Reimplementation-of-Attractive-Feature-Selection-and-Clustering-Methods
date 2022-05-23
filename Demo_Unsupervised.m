%% Add path
addpath('./utils/');
addpath(genpath('./algorithm/'));
rng(495);
%% Load the dataclear X Y;
path='./COIL20.mat';
load(path,'X','Y');
X=X';%% The input dimension is m*n
%% Select a algorithm
% algorithm=@LRLMR;
% algorithm=@AGUFS;
% algorithm=@DSLRL;
% algorithm=@DLUFS; X=double(zscore(X',1))';
algorithm=@SLMEA;
%% Set the hyper-parameters
%%% Notice: you should modify `options`, so as to tune the hyper-parameters
options=struct();
%% Run the algorithm
algorithm(X,Y,options);