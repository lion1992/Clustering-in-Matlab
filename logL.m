function [loglikelihood]=logL(X, gparams,K)

% get the sum of loglikelihood for each iteration;

[N,D]=size(X);
loglikelihood = 0;
for (i = 1:N)
    sumoflikelihood = 0;
    for (k = 1:K)
        % get the parameters of class k;
         weight = gparams(k).weight;
         mean = gparams(k).mean;
         covariance = gparams(k).covariance;
         % get the likelihood of the parameters given ith data point and the weight of class k;
         likelihood = weight * mvnpdf(X(i,:), mean, covariance);
         sumoflikelihood = sumoflikelihood + likelihood;
    end
    % Compute the loglikelihoods of a specific data point
    logl = log(sumoflikelihood);
    loglikelihood = loglikelihood +logl;
end
end