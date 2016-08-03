function [sse] = SSE(X, U);

%  get the total of the K categories' SSE/N from the categorized data points to the mean vectors;  

[K, D] = size(U);
N = 0;
total = 0;
for (k = 1:K)
    % get the subset of X that are labeled class k;
     Xsub = X(X(:, D+1)==k,1:D);
     [n, junk] = size(Xsub);
     centersub = repmat(U(k,:),[n,1]);
     % compute the sum of squared errors between the subset and the mean
     % vector in class k;
     subsse = sum((euclid(Xsub, centersub)).^2);
     total = total + subsse;
     N = N + n;
end
sse = total/N;
end
    
  