function model = svmtrain(X, y, kernel, C, gamma)
% SVMDEMO  Demos the SVM code
%   First generate some training data X, Y by using GENDATA(). Then try
%
%   SVMDEMO(X, Y, 'LINEAR', C) where C is the SVM C parameter.
%
%   SVMDEMO(X, Y, 'RBF', C, GAMMA) where C is the SVM C parameter and
%   GAMMA is the RBF GAMMA parameter
%
%   Restet the random number generator to run twice on the same
%   data. E.g.:
%
%     randn('state', 1) ;
%     [X,y] = gendata(15, 15) ;
%
%     figure(1) ;
%     svmdemo(X, y, 'linear', 1) ;
%
%     figure(2) ;
%     svmdemo(X, y, 'linear', .1) ;
%
%   See also:: SVM(), SVMDEMOALL().
%
%   Author:: Andrea Vedaldi <vedaldi@robots.ox.ac.uk>

% --------------------------------------------------------------------
%                                                       SVM parameters
% --------------------------------------------------------------------

if nargin < 1, kernel = 'linear' ; end
if nargin < 2, C = 1 ; end
if nargin < 3, gamma = 1 ; end

% --------------------------------------------------------------------
%                                                             Training
% --------------------------------------------------------------------

switch kernel
  case 'linear'
    K = X'*X ;
  case 'rbf'
    K = exp(- gamma * pdist2(X,X)) ;
end
model = svm(K,y,C) ;