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

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

X = [ones(m, 1) X];
         
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
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

theta1Col = size(Theta1)(2);
theta2Col = size(Theta2)(2);

ym = [];

for k=1:size(y,1)
    yk = zeros(1,num_labels);
    yk(y(k)) = 1;
    ym = [ym;yk];
end;

a2 = sigmoid(X*Theta1');
a2 = [ones(size(a2, 1), 1) a2];
            

J =  -(1/m)*(sum((log(sigmoid(a2*Theta2')).*ym)(:))+sum((log(1-sigmoid(a2*Theta2')).*(1-ym))(:))) + ...
      (lambda/(2*m))*(sum((Theta1(:,[2:theta1Col]).^2)(:))+sum((Theta2(:,[2:theta2Col]).^2)(:)));


%后向传播
d1Sum = zeros(size(Theta1)(1),size(Theta1)(2));     
d2Sum = zeros(size(Theta2)(1),size(Theta2)(2));

for t=1:m
    fa1 = X(t,:);
    
    fa2 = sigmoid(fa1*Theta1');
    fa2 = [ones(size(fa2, 1), 1) fa2];
    fa3 = sigmoid(fa2*Theta2');  %预测结果

    d3 = fa3 - ym(t,:);
    d2 = (Theta2'*d3').*fa2'.*(1-fa2)'; 
    d2 = d2(2:size(d2)(1));%d2的计算和课程公式略有出入,主要是行向量和列向量的差别,最终
    %d2将会是一个n维向量,n是第二层神经元个数（去掉偏置神经元）
    
    d2Sum = d2Sum + d3'*fa2;
    d1Sum = d1Sum + d2*fa1;
end;

Theta1_grad(:,1) = (1/m)*(d1Sum(:,1));
Theta1_grad(:,[2:size(Theta1_grad)(2)]) = (1/m)*(d1Sum(:,[2:size(d1Sum)(2)])+lambda*Theta1(:,[2:theta1Col]));

Theta2_grad(:,1) = (1/m)*(d2Sum(:,1));
Theta2_grad(:,[2:size(Theta2_grad)(2)]) = (1/m)*(d2Sum(:,[2:size(d2Sum)(2)])+lambda*Theta2(:,[2:theta2Col]));



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
