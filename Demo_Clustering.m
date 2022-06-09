%% Add path
addpath('./utils/');
addpath(genpath('./algorithm/'));
rng(495);
%% Load the dataclear X Y;
path='./COIL20.mat';
load(path,'X','Y');
X=X';%% The input dimension is m*n
%% Select a algorithm
%%% === Clustering ===
algorithm=@rLPP;

%% Set the hyper-parameters
%%% Notice: you should modify `options`, so as to tune the hyper-parameters
options=struct();
%% Run the algorithm
algorithm(X,Y,options);
