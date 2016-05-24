function [c, yhat, g] = multilayer_tanh(W, x, y)
%
% Compute gradient in multilayer rectified linear network
%
%
%

Nl = length(W);

a{1} = x;
for i = 1:Nl-1
   a{i+1} = tanh(W{i}*a{i});
end
nhat = W{Nl}*a{Nl}; % Net input to prediction units (before sigmoid nonlinearity). Saved because it may be useful for computing classifications at different thresholds.
a{Nl+1} = sigmoid(nhat);


yhat = a{Nl+1}; % Model predictions. If all you need is predictions, can stop and return here.

% Cost computation

e = a{Nl+1} - y;
c = sum(sum((1-y).*nhat + log(1 + exp(-nhat))));

if nargin < 3
    return
end
    
delta{Nl+1} = e;

for i = Nl:-1:1
   delta{i} = (W{i}'*delta{i+1}).*(1-a{i}.^2); 
   g{i} = delta{i+1}*a{i}';
end

