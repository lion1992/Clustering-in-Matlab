function [distance] = euclid(x, y);
% get the euclidean distance between x and y;
distance = sqrt(sum((x - y).^2, 2));

end