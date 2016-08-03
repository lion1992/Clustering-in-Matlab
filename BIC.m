function [bic, loglikelihoodseries] = BIC(X, maxK, init_method)
[N,D] = size(X);
bic = zeros([1,maxK]);
loglikelihoodseries = zeros([1,maxK]);

for (i = 1:maxK)
    [~,~, gparams,memberships] =  gaussian_mixture(X,i,init_method,0.000001);
    loglikelihood = logL(X, gparams,i);
    pk = i*(1 + D + D*(D+1)/2);
    bic(i) = loglikelihood - pk/2*log(N);
    loglikelihoodseries(i) = loglikelihood;
end
end
    