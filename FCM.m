% step 1: load data

load('cities_life.mat','categories','names','ratings');

K=10; % 10 clusters;

arrCentroid=zeros(K,9); %center points;

arrayMember=zeros(329, K); % degree of every sample and every cluster.

MAXITER=50;

% step 2: initialize memberships

arrayMember = randi(10, 329, K);

sse = zeros(MAXITER, 1);

for iter=1:MAXITER

% step 3: Calculate the cluster centroids

tempN = zeros(1, 9);

tempD = 0;

oldCentroid = arrCentroid;

for k = 1:K

temp_n = zeros(1, 9);

temp_d = 0;

for i = 1:size(ratings, 1)

temp_n = temp_n + ((arrayMember(i, k) .^ 2) * ratings(i, :) ) ;

temp_d = temp_d + (arrayMember(i, k) .^ 2);

end

arrCentroid(k, :) = temp_n ./ temp_d;

end

% step 4: update membership values

for i = 1:size(ratings, 1)

for k = 1:K

temp_n = sqrt((ratings(i, :) - arrCentroid(k, :)) * (ratings(i, :) - arrCentroid(k, :))');

temp_d = 0;

temp = 0;

for j = 1:K

temp_d = temp_d + sqrt((ratings(i, :) - arrCentroid(j, :)) * (ratings(i, :) - arrCentroid(j, :))');

temp = temp + ((temp_n ./ temp_d) * (temp_n ./ temp_d));

end

arrayMember(k,i) = 1./(temp);

end

end

% step 5: check stop conditions

offset = abs(sum(sum(oldCentroid - arrCentroid)));

% step 6: calculate the sum of squared errors (SSE) as a quantitative metric of your clustering result

for i = 1:size(ratings, 1)

for k = 1:K

sse(iter) = sse(iter) + (arrayMember(i, k) * arrayMember(i, k)) * ((ratings(i, :) - arrCentroid(k, :)) * (ratings(i, :) - arrCentroid(k, :))');

end

end

sse(iter) = sse(iter) / size(ratings, 1);

end

% Plot convergency curve for which the horizontal-direction indicates iteration,
plot(sse);

%and the vertical-direction indicates the objective function F(i.e., SSE).