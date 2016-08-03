function [membership] = getMembership (x, k, gparams, K)
% get the wik, the probability of data point x being in class k;
likelihood = zeros([K,1]);
for (j = 1:K)
    % get the parameters of class j;
    weight = gparams(j).weight;
    mean = gparams(j).mean;
    covariance = gparams(j).covariance;
    % compute and record the likelihood a(j)*p(x|mean(j), covariance(j));
    likelihood(j) = weight * mvnpdf(x, mean, covariance);
end
% get the membership;
membership = likelihood(k)/(sum(likelihood));
end