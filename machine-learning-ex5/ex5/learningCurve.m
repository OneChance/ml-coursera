function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%
%逐渐对增大的样本学习参数
%测试样本数越少,越容易拟合,误差越小,不过随着样本数的增大,误差的递增会越来越不明显
%测试样本数越少,拟合出的直线对于新数据的误差越大,随着测试样本的增加,误差会逐渐减小,不过最终会接近测试样本的误差
%结论:由于参数太少（多项式少）,这是一个欠拟合的情况,所以样本到达一定数量后继续增加,也还是存在不小的误差
% ---------------------- Sample Solution ----------------------

for i=1:m
    [theta] = trainLinearReg(X(1:i,:), y(1:i),lambda);
    error_train(i) = (1/(2*i))*(sum((X(1:i,:)*theta-y(1:i)).^2));
    error_val(i) = (1/(2*(size(Xval)(1))))*(sum((Xval*theta-yval).^2));
end;


% -------------------------------------------------------------

% =========================================================================

end
