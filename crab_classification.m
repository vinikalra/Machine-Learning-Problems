%% step 1  load data 
load('crab.mat', 'X','Y');

n=200;
%split data 
S=randperm(n);
%100 training samples
Xtr=X(:,S(1:100));
Ytr=Y(:,S(1:100));
% 100 testing samples
Xte=X(:,S(101:end));
Yte=Y(:,S(101:end));

%% step 2 further split Xtr/Ytr into two even subsets: use one for training, another for validation.
ntr = size(Xtr,2);
S1 = randperm(ntr);
%training 
Xtrain=Xtr(:,S1(1:50));
Ytrain=Ytr(:,S1(1:50));
% Validation
Xvalid=Xtr(:,S1(51:end));
Yvalid=Ytr(:,S1(51:end));

%% step 3 Model selection over validation set
% consider the parameters C, kernel types (linear, RBF etc.) and kernel
% parameters if applicable. 
error_lin = [];
accuracy_lin = [];
precission_lin = [];
recall_lin = [];

error_rbf = [];
accuracy_rbf = [];
precission_rbf = [];
recall_rbf = [];

Cr = [.05 .1 0.25 0.5 0.75 1] ;
gammar = [.1 0.25 .5  0.75 1] ;

t = 0;

% 3.1 Plot the validation errors while using different values of C ( with other hyperparameters fixed); 
for C = Cr
  t = t + 1;
  model = svmtrain(Xtrain, Ytrain, 'linear', C);
  [Conf_mat, err, a, p, r, pred_y] = svmtest(model, Xtrain, Xvalid, Yvalid, 'linear');
  error_lin(t) = err;
  accuracy_lin(t) = a;
  precession_lin(t,:) = p;
  recall_lin(t,:) = r;
end

t = 0;

% 3.2 Plot the validation errors while using linear, RBF kernel, or Polynomial kernel ( with other hyperparameters fixed); 
for C = Cr
  for gamma = gammar
    t = t + 1;
    model = svmtrain(Xtrain, Ytrain, 'rbf', C, gamma);
    [Conf_mat, err, a, p, r, pred_y] = svmtest(model, Xtrain, Xvalid, Yvalid, 'rbf', gamma);
    error_rbf(t) = err;
    accuracy_rbf(t) = a;
    precession_rbf(t,:) = p;
    recall_rbf(t,:) = r;
  end
end


%% step 4 Select the best model and apply it over the testing subset 
Gamma = 1;
C = 0.5;
error_te = [];
accuracy_te = [];
precission_te = [];
recall_te = [];
%model = svmtrain(Xtr, Ytr, 'rbf', C, Gamma);
%model = svmtrain(Xtrain, Ytrain, 'rbf', C, gamma);
model = svmtrain(Xtrain, Ytrain, 'linear', C);
[Conf_mat, err, a, p, r, pred_y] = svmtest(model, Xtrain, Xte, Yte, 'linear');

%% step 5 evaluate your results with the metrics you have developed in HA3,including accuracy, quantatize your resutls. 
Conf_mat
error_te = err
accuracy_te = a
precession_te = p
recall_te = r
 









