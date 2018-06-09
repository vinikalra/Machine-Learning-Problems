function [X, y]=getDataSet()

% Step 1: Generate data by a module
N = 100; % 1st class contains N objects
alpha = 1.5; % 2st class contains alpha*N ones
sig2 = 1; % assume 2nd class has the same variance as the 1st
dist2 = 4;
%
% % later we move this piece of code in a separate file
% [X, y] = loadModelData(N, alpha, sig2, dist2);
N2 = floor(alpha * N); % calculate the size of the 2nd class
cls1X = randn(N, 2); % generate random objects of the 1st class

% generate a random distance from the center of the 1st class to the center
% of the 2nd
ShiftClass2 = repmat( ...
    dist2 * [sin(pi*rand)  cos(pi*rand)], ...
    [N2,1]); %
% generate random objects of the 2nd class
cls2X = sig2 * randn(N2, 2) + ShiftClass2;
% combine the objects
X = [cls1X; cls2X];
% assign class labels: 0s and 1s
y = [zeros(size(cls1X,1),1); ones(size(cls2X,1),1)];
%end % of module.