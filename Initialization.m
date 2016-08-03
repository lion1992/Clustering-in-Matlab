function [initialparams, initialmemberships] = Initialization(X, K, initialMethod);
initialparams = struct('weight', {}, 'mean', {}, 'covariance', {});
[N, D] = size(X);
initialmemberships = zeros([N, K]);
if(initialMethod == 1)
    % Initialize membership
    for (i = 1:N)
        initialmemberships(i, :) = rand(1,K);
        sumofmemberships = sum(initialmemberships(i,:));
        initialmemberships(i, :) = initialmemberships(i,:)/sumofmemberships;
    end
    for (k = 1:K)
        weight = sum(initialmemberships(:,k))/N;
        mean = ((initialmemberships(:,k))' * X)/(weight * N);
        covariance = 1/(weight * N) * ((repmat(initialmemberships(:,k),[1, D]))' .* (X - repmat(mean,[N,1]))' * (X - repmat(mean,[N,1])));
        
        initialparams(k).weight = weight;
        initialparams(k).mean = mean;
        initialparams(k).covariance = covariance;
    end
    
    
else
    % initialize parameters
    randIndex = randperm(N, K);
    if(initialMethod == 2)
        % initialize the K mean vectors by randomly selecting K data points from X;
         mean = X(randIndex,:);
    elseif(initialMethod == 3)
        % initialize the K mean vectors by Kmeans;
        [mean, ~] = Kmeans(X, K, 1, 0); 
    end

    initialmemberships = zeros([N,K]);
    % compute the rest of the parameters based on the mean vectors;
    for(k = 1:K) 
        initialparams(k).mean = mean(k, :);
        weight = 1/K;
        covariance = 1/N * ((X - repmat(initialparams(k).mean, [N,1])))' * ((X - repmat(initialparams(k).mean, [N,1])));
        initialparams(k).weight = weight;
        initialparams(k).covariance = covariance;
    end
    % compute the membership matrix;
    for (i = 1:N)
        for (k = 1:K)
            initialmemberships(i, k) = getMembership(X(i, :), k, initialparams, K);
        end
    end
end
end
    
    
            
        
    
    
    


    
