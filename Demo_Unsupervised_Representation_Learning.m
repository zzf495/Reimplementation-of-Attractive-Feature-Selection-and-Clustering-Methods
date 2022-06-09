%% Add path
addpath('./utils/');
addpath(genpath('./algorithm/'));
rng(495);
%% Load the dataclear X Y;
path='./COIL20.mat';
load(path,'X','Y');
X=X';%% The input dimension is m*n

%% Split dataset
%%% select 'number' samples from each class as a training set, and use
%%%     as a training set
number=10;
[X1,Y1,X2,Y2] = splitData(X,Y,number);

%% Select a algorithm
%%% === Unsupervised Representation Learning ===
algorithm=@JLRSL;

%% Set the hyper-parameters
%%% Notice: you should modify `options`, so as to tune the hyper-parameters
options=struct();
%% Run the algorithm
algorithm(X1,Y1,X2,Y2,options);


