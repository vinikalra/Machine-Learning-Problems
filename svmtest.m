function [Conf_mat, err, a, p, r, pred_y]  = svmtest(model, Xtrain, X, y, kernel, gamma)
    
  switch kernel
   case 'linear'
      %K = X(:,model.svind)' * X ;
      K = Xtrain(:,model.svind)' * X ;
   case 'rbf'
      K = exp(- gamma * pdist2(Xtrain(:,model.svind), X)) ;
  end
  
  pred = model.alphay(model.svind)' * K + model.b ;
  pred_y = sign(pred);
  
  Conf_mat = confusionmat(y, pred_y);
  
   a= sum(diag(Conf_mat)) / sum(sum(Conf_mat));
   p = diag(Conf_mat) ./ sum(Conf_mat,2);
   r = diag(Conf_mat) ./ sum(Conf_mat,1)';
   err = vpa((numel(y) - sum(diag(Conf_mat))) / sum(sum(Conf_mat)));
  