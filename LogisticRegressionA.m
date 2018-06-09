%% step 1: generate dataset that includes both positive and negative samples,
% where each sample is described with two features. 
%250 samples in total.

[X,y]=getDataset(); % note that y contains only 1s and 0s,
size(X)
 
%plotting all samples
    idx1 = find(y == 0); % object indices for the 1st class
    idx2 = find(y == 1);
    h=subplot(1,3,1);  hold on;
    % no more variables are needed
    plot(X(idx1,1), X(idx1,2), 'r*');
    plot(X(idx2,1), X(idx2,2), 'b*');
    %axis tight
    xlabel('x_1');
    ylabel('x_2');
    title('All samples');
    
%number of training samples
ntrain=120;% 

 % Randomly pick up 100 samples for training and use
 % the rest for teting. 
trainingSet = randperm(size(X,1),ntrain);
trainX=X(trainingSet, :); % training samples,
trainy=y(trainingSet); % labels of training samples

testSet =setdiff(1:size(X,1),trainingSet);
testX=X(testSet,:); % testing samples
testy=y(testSet); % labels of testing samples
 
 % plot the samples you have pickup for training, check to confirm that both negative
 % and positive samples are included. 
 
    idx1 = find(trainy == 0); % object indices for the 1st class
    idx2 = find(trainy == 1);
    h =subplot(1,3,2); hold on;
    % no more variables are needed
    plot(trainX(idx1,1), trainX(idx1,2), 'r*');
    plot(trainX(idx2,1), trainX(idx2,2), 'b*');
    %axis tight
    
    xlabel('x_1');
    ylabel('x_2');
    title('training samples');

    idx1 = find(testy == 0); % object indices for the 1st class
    idx2 = find(testy == 1);
    % no more variables are needed
    h=subplot(1,3,3); hold on; 
    plot(testX(idx1,1), testX(idx1,2), 'r*');
    plot(testX(idx2,1), testX(idx2,2), 'b*');
    %axis tight
    xlabel('x_1');
    ylabel('x_2');
    title('testing samples');    
    
%% step 2: train logistic regression model with the gradient descent method 

bHat = glmfit(trainX,trainy,'binomial');

%% step 3: Use the model to get class labels of testing samples.
    % with the learned model, apply the logistic model over testing
    % samples; hatProb is the probability of belonging to the class 1.
    hatProb = 1 ./ (1+exp(-[ones(size(testX,1),1),testX]*bHat)); % variant of classification
 
 % predict the class labels with a threshold
haty = (hatProb>=0.5); 
%% step 4: evaluation
 
   %compare haty and testy to calculate average error and standard deviation  %   
   avgErr=mean(abs(haty-testy));
   stdErr=std(abs(haty-testy));
 

 fprintf('average error:%f (%f)\n', avgErr, stdErr);
 