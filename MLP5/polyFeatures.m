function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 
iter = 1;
pow_vec = zeros(p, 1);
for i = 1:p;
  pow_vec(i) = iter;
  iter = iter + 1;
end;

X_mat = ones(length(X), p).*X;
for n = 1:length(X)
  for l = 1:p
    X_poly(n,l) = X_mat(n,l).^pow_vec(l);
  
end;


% =========================================================================

end
