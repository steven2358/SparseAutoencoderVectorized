%% CS294A/CS294W Programming Assignment Starter Code
% Perform vectorized training
% http://ufldl.stanford.edu/wiki/index.php/Vectorization

close all;
clear all;

%% PARAMETERS

visibleSize = 28*28; % number of input units 
hiddenSize = 196; % number of hidden units 
sparsityParam = 0.1; % desired average activation of the hidden units
lambda = 3E-3;  % weight decay parameter       
beta = 3; % weight of sparsity penalty term       

addpath minFunc/
options.Method = 'lbfgs'; % optimization algorithm
options.maxIter = 400; % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

%% LOAD DATA

patches = get_data();
% display_network(patches(:,randi(size(patches,2),200,1)),8);

%% TRAINING

%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, visibleSize);

% Train autoencoder with minFunc (L-BFGS).
% Generally, for minFunc to work, you need a function pointer with two
% outputs: the function value and the gradient.
tic
[opttheta, cost] = minFunc(@(p) sparseAutoencoderCostVectorized(...
    p, visibleSize, hiddenSize, lambda, sparsityParam, beta, patches), ...
    theta, options);
toc
%% OUTPUT

W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
display_network(W1', 12);

print -djpeg weights.jpg   % save the visualization to a file 
