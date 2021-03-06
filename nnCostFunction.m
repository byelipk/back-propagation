function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight
% matrices for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% Expand the output values into a matrix of single values
y_matrix = eye(num_labels)(y, :);

% FORWARD PROPAGATION

% Add the bias unit and rename to 'a1' - we can think of this as representing
% the activation of our input layer.
a1 = [ ones(size(X, 1), 1), X ];

% z2 is the linear combination of Theta1 and the input values. To
% compute this using matrix multiplication we need to take the transpose of
% a1 (so the number of columns in Theta1 equals the number of rows in a1).
z2 = Theta1 * a1';

% We obtain the activation of layer 2 by applying z2 to the sigmoid function.
% We also take the transpose to make matrix algebra easier. Then we add the
% bias unit to complete the construction of activation layer 2.
a2 = sigmoid(z2);
a2 = a2';
a2 = [ones(size(a2, 1), 1), a2];

% z3 is the linear combination of Theta2 and activation layer 2.
z3 = Theta2 * a2';

% We obtain the activation of the output layer (layer 3) by applying z3
% to the sigmoid function. Just repeat the steps we used to compute a2.
a3 = sigmoid(z3);
a3 = a3';

% Now we can compute the cost function using a3, y_matrix, and m.

% Unregularized cost
J = (1/m) * sum(sum((-y_matrix .* log(a3)) - ((1 - y_matrix) .* log(1 - a3))));

% Regularized cost - exclude the first column of bias units
J = J + (lambda / (2 * m)) * (
  sum(sum(Theta1(:, 2:end).^2)) +
  sum(sum(Theta2(:, 2:end).^2))
);

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial
%         derivatives of the cost function with respect to Theta1 and Theta2
%         in Theta1_grad and Theta2_grad, respectively. After implementing
%         Part 2, you can check that your implementation is correct by running
%         checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%

% m = the number of training examples
% n = the number of training features, including the initial bias unit.
% h = the number of units in the hidden layer - NOT including the bias unit
% r = the number of output classifications

% BACK PROPAGATION

% NOTE
% We've already computed the activations (z2, a2, z3, a3) for layers 2 and 3
% so we'll reuse them for the backprop implementation.
d3 = a3 - y_matrix;
d2 = (Theta2(:, 2:end)' * d3')' .* sigmoidGradient(z2)';

Delta1 = d2' * a1;
Delta2 = d3' * a2;

Theta1_grad = Delta1 .* (1/m);
Theta2_grad = Delta2 .* (1/m);


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1(:, 1) = 0;
Theta2(:, 1) = 0;

Theta1_grad = Theta1_grad + (Theta1 * (lambda / m));
Theta2_grad = Theta2_grad + (Theta2 * (lambda / m));


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
