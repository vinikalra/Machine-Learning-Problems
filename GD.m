function [theta, arrCost] = GD(X, theta, y, alpha, numIters)

% Gradient Descent Method

m = length(y);

arrCost = zeros(numIters, 1);

thetaLen = length(theta);

tempVal = theta;

for iter=1:numIters

%Update theta, i.e., the parameters to estimate.

temp = (X*theta - y);

for i = 1:thetaLen

tempVal(i) = sum(temp.*X(:,i));

end

theta = theta - (alpha/m)*tempVal;

%calculate the current cost with the present theta;

arrCost(iter) = (1/(m)).*sum((y-X*theta).^2);

end