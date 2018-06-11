

%% Step 1: generating training data
% Make training data of two classes "red" and "blue"
 % with 2 features for each sample (the position  x and y).
  angle=rand(200,1)*2*pi; l=rand(200,1)*40+30; blue=[sin(angle).*l cos(angle).*l];
  angle=rand(200,1)*2*pi; l=rand(200,1)*40;    red=[sin(angle).*l cos(angle).*l];

 % All the training data
  datafeatures=[blue;red];
  dataclass(1:200)=-1; dataclass(201:400)=1;

 % Show the data: include these figures in your report. 
  figure, subplot(2,2,1), hold on; axis equal;
  plot(blue(:,1),blue(:,2),'b.'); plot(red(:,1),red(:,2),'r.');
  title('Training Data');
  
 %% step-2: training of a strong classifier using Adaboost
 %function [estimateclasstotal,model]=adaboost(mode,datafeatures,dataclass_or_model,itt)
 
        itt=80; %The number of training itterations
         
        % data classes
        dataclass=dataclass(:);
        
        %Output: a set of weak classifiers; 
        model=struct;
        
        % Weight of training samples, first every sample is even important
        % (same weight)
        D=ones(length(dataclass),1)/length(dataclass);
        
        % This variable will contain the results of the single weak
        % classifiers weight by their alpha
        estimateclasssum=zeros(size(dataclass));
        
        % Calculate max min of the data
        boundary=[min(datafeatures,[],1) max(datafeatures,[],1)];
        
        % Do all model training itterations
        for t=1:itt
            % Find the best treshold to separate the data in two classes;
            % It is noteworthy that D will vary over iterations; 
            %% step 2.1, select the optimal weak classifiers
            [estimateclass,err,h] = WeightedThresholdClassifier(datafeatures,dataclass,D); 
            % h: struct variable, the selected weak classifier; 
            % err: error rate; 
            
            

            %% step 2.2  Weak classifier influence on total result is based on the current
            % classification error
            
             eps = 1e-16;
            alpha = 0.5 * log((1 - err) / max(err, eps));
            
            % Store the model parameters
            model(t).alpha = alpha;
            model(t).dimension=h.dimension;
            model(t).threshold=h.threshold;
            model(t).direction=h.direction;
            model(t).boundary = boundary;
            %% step 2.3 We update D so that wrongly classified samples will have more weight
            D = D .* exp(-model(t).alpha.*dataclass.*estimateclass);
            D = D ./ sum(D);
            
            % Calculate the current error of the cascade of weak
            % classifiers
            estimateclasssum=estimateclasssum +estimateclass*model(t).alpha;
            classestimate=sign(estimateclasssum);
            model(t).error=sum(classestimate~=dataclass)/length(dataclass);
            if(model(t).error==0), break; end
        end
        % end of training 
 
 %[classestimate,model]=adaboost('train',datafeatures,dataclass,80);

 % Visualization of training results
 % Show results
  blue=datafeatures(classestimate==-1,:); red=datafeatures(classestimate==1,:);
  I=zeros(161,161);
  for i=1:length(model)
      if(model(i).dimension==1)
          if(model(i).direction==1), rec=[-80 -80 80+model(i).threshold 160];
          else rec=[model(i).threshold -80 80-model(i).threshold 160 ];
          end
      else
          if(model(i).direction==1), rec=[-80 -80 160 80+model(i).threshold];
          else rec=[-80 model(i).threshold 160 80-model(i).threshold];
          end
      end
      rec=round(rec);
      y=rec(1)+81:rec(1)+81+rec(3); x=rec(2)+81:rec(2)+81+rec(4);
      I=I-model(i).alpha; I(x,y)=I(x,y)+2*model(i).alpha;    
  end
 subplot(2,2,2), imshow(I,[]); colorbar; axis xy;
 colormap('jet'), hold on
 plot(blue(:,1)+81,blue(:,2)+81,'bo');
 plot(red(:,1)+81,red(:,2)+81,'ro');
 title('Training Data classified with adaboost model');

 % Show the error verus number of weak classifiers
 error=zeros(1,length(model)); for i=1:length(model), error(i)=model(i).error; end 
 subplot(2,2,3), plot(error); title('Classification error versus number of weak classifiers');

 %% step 3. testing of new samples 
   % Make some test data
  angle=rand(200,1)*2*pi; l=rand(200,1)*70; 
  testdata=[sin(angle).*l cos(angle).*l]; % feature matrix for the testing data; 

 % Classify the testdata with the trained model
 % testclass=adaboost('apply',testdata,model);
    datafeatures=testdata;
  % Apply Model on the test data
         
        % Limit datafeatures to orgininal boundaries
        if(length(model)>1);
            minb=model(1).boundary(1:end/2);
            maxb=model(1).boundary(end/2+1:end);
            datafeatures=bsxfun(@min,datafeatures,maxb);
            datafeatures=bsxfun(@max,datafeatures,minb);
        end
    
        % Add all results of the single weak classifiers weighted by their alpha 
        estimateclasssum=zeros(size(datafeatures,1),1);
        for t=1:length(model);
            %% step 3.1 applying weak classifiers to 
             estimateclasssum = estimateclasssum + model(t).alpha*ApplyClassTreshold(model(t),datafeatures);
                       
        end
        % If the total sum of all weak classifiers
        % is less than zero it is probablly class -1 otherwise class 1;
        testclass=sign(estimateclasssum);
 
 % visualization fo testing result
  blue=testdata(testclass==-1,:); red=testdata(testclass==1,:);

 % Show the data
  subplot(2,2,4), hold on
  plot(blue(:,1),blue(:,2),'b*');
  plot(red(:,1),red(:,2),'r*');
  axis equal;
  title('Test Data classified with adaboost model');
