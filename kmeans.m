% K-means

% step 1: load data
load('cities_life.mat','categories','names','ratings'); %feature vectors in ratings
K=10; % 10 clusters; 

arrCentroid=zeros(K,9); %center points; 
arrMember=zeros(1,329); % cluster index of every sample, 1-K;
MAXITER=50;

data_dim = length(ratings(1,:));

nbData = length(ratings(:,1));

% step 2: initialize centroids 

minData = min(ratings);

maxData = max(ratings);

difference = maxData - minData ;

arrCentroid=ratings(1:K,:); % use the top K samples; 

% no stopping at start

steps = 1.;

for iter=1:MAXITER
% step 3: assign every sample to its closed cluster. Use eucliden distance 
while steps > 0.0
    
    % E-Step

assignment = [];

% assign each datapoint to the closest centroid

for d = 1 : length( ratings(:, 1) );

minDifference = ( ratings( d, :) - arrCentroid( 1,:) );

minDifference = sqrt(minDifference * minDifference');

curAssignment = 1;

for c = 2 : K;

diff2c = ( ratings( d, :) - arrCentroid( c,:) );

diff2c = sqrt(diff2c * diff2c');

if( min_diff >= diff2c)

curAssignment = c;

minDifference = diff2c;

end

end

% assign the d-th dataPoint

assignment = [ assignment; curAssignment];

end

% for the stoppingCriterion

oldPositions = arrCentroid;

% step 4: re-calculate the centroid of every cluster. 

arrCentroid = zeros(K, data_dim);

pointsInCluster = zeros(K, 1);

for d = 1: length(assignment);

arrCentroid( assignment(d),:) = arrCentroid( assignment(d),:) + ratings(d,:);

pointsInCluster( assignment(d), 1 ) = pointsInCluster( assignment(d), 1 ) + 1;

end

for c = 1: K;

if( pointsInCluster(c, 1) ~= 0)

arrCentroid( c , : ) = arrCentroid( c, : ) / pointsInCluster(c, 1);

else

% set cluster randomly to new position

arrCentroid( c , : ) = (rand( 1, data_dim) .* difference) + minData;

end

end

% step 5: check stop conditions

steps = sum (sum( (arrCentroid - oldPositions).^2 ) );

end

end
% step 6: calculate the sum of squared errors (SSE) as a quantitative
idx_arr = {[],[],[],[],[],[],[],[],[],[]};

for i = 1:10

for j=1:length(assignment)

if assignment(j) == i

idx_arr{i} = [idx_arr{i}, j];

end

end

end

sse = zeros(10,1);

totSSE = 0;

for i = 1:10

for j = idx_arr{i}

sse(i) = sse(i) + ((ratings(j,:) - arrCentroid(i)) * (ratings(j,:) - arrCentroid(i))');

end

totSSE = totSSE + sse(i);

end

plot(sse);
