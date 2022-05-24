%% Add path
addpath('./utils/');
addpath(genpath('./algorithm/'));
rng(495);
%% Load the dataclear X Y;


%%% Load COIL20
% path='./COIL20.mat';
% load(path,'X','Y');
% X=X';%% The input dimension is m*n


%%% Load AR_Face_img
% path='./AR_Face_img.mat';
% clear AllSet;
% load(path,'AllSet');
% XY=AllSet;
% X=XY.X;
% Y=XY.y;


%% Data process
%%% try `zscore` or `normr`
% X=L2Norm(X')';
% X=double(zscore(X',1))';
% X=normr(X')';


%% Split dataset
%%% select 'number' samples from each class as a training set, and use
%%%     as a training set
number=10;
[X1,Y1,X2,Y2] = splitData(X,Y,number);


%% Select a algorithm
% algorithm=@RSLDA;
% algorithm=@DSDPL; X=double(zscore(X',1))';


%% Set the hyper-parameters
%%% Notice: you should modify `options`, so as to tune the hyper-parameters
options=struct();


%% Run the algorithm
algorithm(X1,Y1,X2,Y2,options);
