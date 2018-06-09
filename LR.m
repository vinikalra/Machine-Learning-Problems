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

%
X; y; % note that y contains only 1s and 0s
idx1 = find(y == 0); % object indices for the 1st class
idx2 = find(y == 1);
% no more variables are needed

%Step 2:  preliminary plot
h = figure; hold on
plot(X(idx1,1), X(idx1,2), 'r*');
plot(X(idx2,1), X(idx2,2), 'b*');
axis tight
xlabel('x_1');
ylabel('x_2');
% close(h);

% step 3: Use a logistic regression as a classification algorithm to classify objects
bHat = glmfit(X,y,'binomial');
% step 4: We have the model y = 1/(1+exp(-Xb)) and its parameters bHat. Use the model to get the estimations of class labels.
yHat = 1./(1+exp( -[ones( size(X,1),1 ), X] *bHat)); % variant of classification
%yHat = glmval(bHat, X, 'logit'); % variant of classification

% % formed as an inline function
% classify = inline( '1./(1+exp( -[ones( size(X,1),1 ), X] *b))', 'b', 'X');
% % separation hyperplane, formed as a function (here it is a line)
% separateXLim = inline( '(-b(1)- YLim*b(3))/b(2)', 'b','YLim');
% 
% % example of classification model usage
% yHat = classify(bHat,X);



% Step 4: the objects could be surrounded by circles
%figure (2); hold on;
idx1 = find(yHat < 1/2); % object indices for the 1st class
idx2 = find(yHat >= 1/2);
plot(X(idx1,1), X(idx1,2), 'ro');
plot(X(idx2,1), X(idx2,2), 'bo');

% or separated by plane
 %plot(separateXLim(bHat,YLim), YLim, 'b-');

return; 

%To plot 3D surface of probability we should compute the probability function in each point of the X1 by X2 - plane.

% to do that create a grid
GRIDSIZE = 10;
linX1 = linspace(min(X(:,1)), max(X(:,1)), GRIDSIZE); % grid of feature values
linX2 = linspace(min(X(:,2)), max(X(:,2)), GRIDSIZE);
[grdX1 grdX2] = meshgrid(linX1, linX2); % cartesian product of two grid sets

grdX1 = grdX1(:); % vectorize the obtained matrices
grdX2 = grdX2(:);

Tri = delaunay(grdX1,grdX2); % make Delauney triangulation
trisurf(Tri,grdX1,grdX2,classify(bHat,[grdX1 grdX2])-1/2,...
        'FaceColor','red','EdgeColor','none');
camlight left; lighting phong; % alpha(0.8);
% the plane was shifted down to 1/2

% to save the picture use the figure handle h
% saveas(h, 'demoDataGen_saved.png', 'png');
% saveas(h, 'demoDataGen.eps', 'psc2');
%close(h);

