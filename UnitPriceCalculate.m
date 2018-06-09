%% Ground-truth Cashier
gdUnitPrice =[20 25 8]; % for fish, chip, and ketchup, respectively

%% Your iterative methods
% step 1: initialize your guess on the unit prices of fish, chip and
% ketchup.
estUnitPrice =[10 10 10]; %initial unit prices.
%set your own stopping conditions and learning rate

%condition 1: maximal iterations, stop.
MAX_ITERATION=160;

%condition 2: if the difference between your prediction and the cashier's
%price is smaller than a threshold, stop.
MIN_DELTA= 0.02;
% learning rate
alpha=1.22e-3;

% step 2: iterative method
for i=1:MAX_ITERATION
% order a meal (simulating)
meal=round(rand(1,3)*10);
%acquire cashier's price;
cashierPrice= sum(meal.*gdUnitPrice);
%calculate delta values
estTotalPrice= sum(meal.*estUnitPrice);
%update unit prices
delta= estTotalPrice - cashierPrice;
estUnitPrice=estUnitPrice-(alpha.*(delta.*meal));
%check stop conditions
if abs(delta) < MIN_DELTA
break;
end
fprintf('iteration:%d, delta:%f \n',i, abs(delta));
end

disp(estUnitPrice);
% step 3: evaluation
error=mean(abs(estUnitPrice-gdUnitPrice));
fprintf('estimation error:%d \n',error);