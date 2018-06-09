%test multiple learning rates and report their convergence curves.
alpha=0.09;
MAX_ITER=500;
load('sat.mat','sat'); % three columns: MATH SAT, VERB SAT, UNI. GPA

sat(:,1)=(sat(:,1)+min(sat(:,1)))./ (max(sat(:,1)).^2);
sat(:,2)=(sat(:,2)+min(sat(:,2)))./ (max(sat(:,2)).^2);
sat(:,3)=(sat(:,3)+min(sat(:,3)))./ (max(sat(:,3)).^2);

%training data;
satTrain=sat(1:60,:);
% testing data;
satTest=sat(61:end,:);
theta=zeros(3,1);

% call the GD algorithm
[theta, arrCost] = GD([ones(60,1) satTrain(:,1:2)], theta, satTrain(:,3), alpha, MAX_ITER);

%visualize the convergence curve
plot(1:length(arrCost),arrCost);
xlabel('iteration');
ylabel('cost');
tVal=[ones(length(satTest),1) satTest(:,1:2)]*theta;
tError=sqrt((tVal-satTest(:,3)).^2);
fprintf('results: %f (%f) \n',mean(tError),std(tError));