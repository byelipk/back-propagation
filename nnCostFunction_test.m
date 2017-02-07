il = 2;              % input layer
hl = 2;              % hidden layer
nl = 4;              % number of labels
nn = [ 1:18 ] / 10;  % nn_params
X = cos([1 2 ; 3 4 ; 5 6]);
y = [4; 2; 3];

lambda = 4;

Theta1 = reshape(nn(1:hl * (il+ 1)), hl, (il + 1));
Theta2 = reshape(nn((1 + (hl * (il + 1))):end), nl, (hl+ 1));

% Test Case with Regularization
[J grad] = nnCostFunction(nn, il, hl, nl, X, y, lambda)

% J = 19.474
% grad =
% 0.76614
% 0.97990
% 0.37246
% 0.49749
% 0.64174
% 0.74614
% 0.88342
% 0.56876
% 0.58467
% 0.59814
% 1.92598
% 1.94462
% 1.98965
% 2.17855
% 2.47834
% 2.50225
% 2.52644
% 2.72233


% Test Case without Regularization
[J grad] = nnCostFunction(nn, il, hl, nl, X, y, 0)

% J =  7.4070
% grad =
%    0.766138
%    0.979897
%   -0.027540
%   -0.035844
%   -0.024929
%   -0.053862
%    0.883417
%    0.568762
%    0.584668
%    0.598139
%    0.459314
%    0.344618
%    0.256313
%    0.311885
%    0.478337
%    0.368920
%    0.259771
%    0.322331
