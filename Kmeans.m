function [U, Xnew] = Kmeans(X, K, R, plotflag);
% U is a K x D matrix, the kth row is a vector defining the center of the kth cluster;
%
% Xnew is a N x (d+1) matrix, the (d+1)th column is the indicator for the
% class whose center the data point is closest to.

% plotflag default is 0

if(nargin < 4 || isempty('plotflag'))
    plotflag = 0;
end

[N, D] = size(X);

distance = zeros([N,K]);
Umin = ones([K, D]);
Xmin = [X, ones(N,1)];
counter=0;
DistanceforPlot = [];
for (r = 1:R)
    % initialize the mean vectors for the K classes for the rth turn;
    initIndex = randperm(N, K);
    u = X(initIndex,:);
    c = zeros([N,1]);
    unew = u.*2;
    done = false; 
    while (done == false)
        % compute the distances from the data points to the K mean vectors
        % in turn;
        for (k = 1:K)
             center = repmat(u(k,:), [N,1]);
            distance(:,k) = (euclid(X, center));
        end  
         % get the indicating vector as for which column (class) has the
         % smallest distance from a specific data point;
         [~, I] = min(distance, [], 2);
         c = I;
         % combining the X matrix with the indicating vector;
         Xc = [X,c];
         % update the mean vectors for the K classes using the corresponding subsets of X in turn;  
         for (k = 1:K)
             Xsub = Xc(Xc(:, D+1)==k,1:D);
             unew(k,:) = mean( Xsub, 1);
         end
         % check if the stopping criterion is satisfied;
         if (max((unew-u) * (unew-u)')<0.01)
             done = true;
         else
             u = unew;
         end
         %  compute the total of the sum of squared errors from data ponits 
         %  in X to their assigned mean vectors; 
         junkDistance = SSE(Xc, unew);
         % record the SSE of this iteration;
         DistanceforPlot = [DistanceforPlot, junkDistance];
         counter = counter+1;
    end
    % compute the SSE from the turn that gives the smallest SSE, the mean 
    % vectors produced by that turn are labled Umin;;
    originUminDistance = SSE(Xmin, Umin);
    % check if the SSE of this turn is smaller than the turn that gives the
    % previous smallest SSE;
    % if yes, then set the mean vectors of this turn as Umin;
    if(junkDistance < originUminDistance)
        Xmin = Xc;
        Umin = unew;
    end

end

U = Umin;
Xnew = Xmin;
    % plot the SSE vs. Iteration;
    if(plotflag == 1)
        plot(1:counter,DistanceforPlot, 'b-');
        hold on;
        ylabel('Sum of Squared Errors');
        xlabel ('Iteration');
        title('SSE vs. Iteration');
        hold off
    end
    
end
             
          
        
    
   
        
        
     
    
    
    
